"""Controlled Transformer trial on course train/validation splits only."""
import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from train import (CaptionDataset, build_vocab, collate_train, evaluate,
                   get_transforms, read_labels, PAD)
from transformer_model import TransformerCaptioner


def atomic_save(payload, path, inference=False):
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp)
    if inference and temp.stat().st_size >= 200_000_000:
        raise ValueError('Whole inference checkpoint exceeds 200 MB')
    temp.replace(path)


def run(args):
    torch.set_num_threads(8)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('This trial requires the available CUDA GPU')
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    root = Path(__file__).resolve().parents[1]
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    history = out / 'training_history.csv'
    if not args.resume and (history.exists() or (out / 'best_model.pt').exists()):
        raise FileExistsError('Use a new trial directory or --resume')
    rows = read_labels(root / 'Training_Labels.csv')
    val_rows = read_labels(root / 'Validation_Labels.csv')
    assert len(rows) == 12000 and len(val_rows) == 2500
    vocab = build_vocab(rows)
    train_t, eval_t = get_transforms()
    gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(CaptionDataset(rows, root/'Training', vocab, train_t, 34),
        batch_size=args.batch_size, shuffle=True, generator=gen, num_workers=args.workers,
        pin_memory=True, collate_fn=collate_train)
    val_loader = DataLoader(CaptionDataset(val_rows, root/'Validation', vocab, eval_t, 34),
        batch_size=48, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_train)
    model = TransformerCaptioner(len(vocab), pretrained=not args.resume).to(device)
    for p in model.encoder.parameters():
        p.requires_grad = False
    optimizer = torch.optim.AdamW([
        {'params': [p for name,p in model.named_parameters() if not name.startswith('encoder.')], 'lr':3e-4},
        {'params': model.encoder.parameters(), 'lr':2e-5},
    ], weight_decay=1e-4)
    steps = len(train_loader)
    total = args.epochs * steps
    warmup = steps
    def factor(step):
        if step < warmup:
            return (step+1)/warmup
        progress = min(1., (step-warmup)/max(1,total-warmup))
        return 0.1 + 0.9*0.5*(1+math.cos(math.pi*progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.05)
    best, stale, start = -math.inf, 0, 1
    if args.resume:
        state = torch.load(out/'last_training.pt', map_location='cpu', weights_only=True)
        expected = {k:v for k,v in vars(args).items() if k != 'resume'}
        if state['args'] != expected or state['vocab'] != vocab:
            raise ValueError('Resume must keep the original configuration and vocabulary')
        model.load_state_dict(state['state_dict'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        best, stale, start = state['best'], state['stale'], state['epoch']+1
        random.setstate(state['random'])
        ns = state['numpy']
        np.random.set_state((ns[0],np.array(ns[1],dtype=np.uint32),ns[2],ns[3],ns[4]))
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        gen.set_state(state['loader_rng'])
    else:
        metadata = dict(args=vars(args), model=model.config,
            train_rows=len(rows), validation_rows=len(val_rows), vocab_size=len(vocab),
            torch_version=str(torch.__version__), gpu=torch.cuda.get_device_name(),
            parameters=sum(p.numel() for p in model.parameters()),
            recipe='AMP; frozen BN; final ResNet stage unfrozen epoch3; warmup1epoch+cosine; beam3 selection')
        (out/'config.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        for source in ['train_transformer.py','transformer_model.py','train.py','decoding_search.py','model_io.py']:
            shutil.copyfile(Path(__file__).parent/source,out/('source_'+source))
        with history.open('w',newline='',encoding='utf-8') as f:
            csv.writer(f).writerow(['epoch','token_loss','validation_similarity','minutes','peak_gpu_bytes'])
    if stale >= args.patience:
        print('Saved run already reached early stopping', flush=True)
        return
    print(f'Transformer trial: {len(vocab)} tokens, {sum(p.numel() for p in model.parameters()):,} parameters',flush=True)
    for epoch in range(start,args.epochs+1):
        if epoch >= 3:
            for p in model.encoder[-1].parameters():
                p.requires_grad = True
        model.train()
        model.encoder.eval()  # Keep pretrained BN statistics fixed, including layer4.
        torch.cuda.reset_peak_memory_stats()
        began = time.monotonic()
        loss_sum, tokens = 0.,0
        for step,(images,captions,_) in enumerate(train_loader,1):
            images,captions = images.to(device,non_blocking=True),captions.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                logits = model(images,captions)
                target = captions[:,1:]
                loss = criterion(logits.reshape(-1,len(vocab)),target.reshape(-1))
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                scheduler.step()
            count = target.ne(PAD).sum().item()
            loss_sum += loss.item()*count
            tokens += count
            if step % 100 == 0 or step == steps:
                print(f'epoch {epoch:02d}/{args.epochs} step {step}/{steps} loss={loss_sum/tokens:.4f}',flush=True)
        score,prediction = evaluate(model,val_loader,vocab,val_rows,root,device,True,24,
            out/'validation_predictions.csv',decode_policy='beam3',length_alpha=0.6)
        improved = score > best
        best = max(best,score)
        stale = 0 if improved else stale+1
        checkpoint = dict(state_dict=model.state_dict(),vocab=vocab,config=model.config,
                          epoch=epoch,validation_similarity=score)
        atomic_save(checkpoint,out/'last_model.pt',inference=True)
        if improved:
            atomic_save(checkpoint,out/'best_model.pt',inference=True)
            shutil.copyfile(prediction,out/'validation_predictions_best.csv')
        ns = np.random.get_state()
        train_state = dict(checkpoint,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            scaler=scaler.state_dict(),best=best,stale=stale,
            args={k:v for k,v in vars(args).items() if k!='resume'},
            random=random.getstate(),numpy=(ns[0],ns[1].tolist(),ns[2],ns[3],ns[4]),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),loader_rng=gen.get_state())
        atomic_save(train_state,out/'last_training.pt')
        elapsed=(time.monotonic()-began)/60
        peak=torch.cuda.max_memory_allocated()
        with history.open('a',newline='',encoding='utf-8') as f:
            csv.writer(f).writerow([epoch,loss_sum/tokens,score,elapsed,peak])
        print(f'epoch {epoch:02d}: similarity={score:.6f}, best={best:.6f}, minutes={elapsed:.2f}, peak={peak/1e9:.2f} GB',flush=True)
        if stale >= args.patience:
            print('Early stopping',flush=True)
            break
    result=dict(best_validation_similarity=best,epochs_completed=epoch,
                checkpoint_bytes=(out/'best_model.pt').stat().st_size,
                decode_policy='beam3',length_alpha=0.6,seed=args.seed)
    (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',default=str(Path(__file__).parent/'runs/resnet50_transformer_v1'))
    parser.add_argument('--epochs',type=int,default=16)
    parser.add_argument('--patience',type=int,default=4)
    parser.add_argument('--batch-size',type=int,default=24)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--seed',type=int,default=2205)
    parser.add_argument('--resume',action='store_true')
    run(parser.parse_args())
