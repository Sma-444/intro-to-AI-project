"""Preflight actual mask semantics, gradients, beam batching, and checkpoint reload."""
import json
import argparse
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformer_model import TransformerCaptioner
from model_io import model_from_checkpoint
from train import CaptionDataset, read_labels, build_vocab, collate_train, get_transforms, PAD
from decoding_search import search_features


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--backbone',choices=['resnet50','dinov2_small'],default='resnet50')
    parser.add_argument('--visual-tail-blocks',type=int,default=2)
    parser.add_argument('--image-size',type=int,choices=[224,336],default=224)
    parser.add_argument('--decoder-layers',type=int,choices=[2,4],default=2)
    args=parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(2205)
    root = Path(__file__).resolve().parents[1]
    out = root/'captioning/runs'/('transformer_preflight' if args.backbone=='resnet50' else 'dinov2_preflight')
    if args.visual_tail_blocks != 2:
        out=out.with_name(out.name+f'_tail{args.visual_tail_blocks}')
    if args.image_size != 224:
        out=out.with_name(out.name+f'_size{args.image_size}')
    if args.decoder_layers != 2:
        out=out.with_name(out.name+f'_decoder{args.decoder_layers}')
    out.mkdir(exist_ok=True)
    rows=read_labels(root/'Training_Labels.csv')
    vocab=build_vocab(rows)
    preprocessing='resize256_centercrop224' if args.image_size==224 else 'scaled_centercrop'
    _, transform=get_transforms(preprocessing,args.image_size)
    from PIL import Image
    with Image.open(root/'Training'/rows[0]['image_filename']) as image:
        image=image.convert('RGB')
        _, legacy=get_transforms()
        _, scaled224=get_transforms('scaled_centercrop',224)
        assert torch.equal(legacy(image),scaled224(image)), 'Legacy preprocessing changed'
    batch=next(iter(DataLoader(CaptionDataset(rows[:24],root/'Training',vocab,transform,34),
        batch_size=24,collate_fn=collate_train)))
    images,captions=batch[0].cuda(),batch[1].cuda()
    model=TransformerCaptioner(len(vocab),backbone_name=args.backbone,
                               decoder_layers=args.decoder_layers).cuda().eval()
    assert len(model.decoder.layers)==args.decoder_layers
    model.config.update(preprocessing=preprocessing,image_size=args.image_size)
    if args.backbone=='dinov2_small':
        from safetensors.torch import load_file
        source=load_file(str(root/'captioning/.weights/dinov2-small/model.safetensors'))
        # Transformers 5 renamed the four attention projections on load.
        replacements={'.attention.attention.query.':'.attention.q_proj.',
                      '.attention.attention.key.':'.attention.k_proj.',
                      '.attention.attention.value.':'.attention.v_proj.',
                      '.attention.output.dense.':'.attention.o_proj.'}
        for old,new in replacements.items():
            source={name.replace(old,new):tensor for name,tensor in source.items()}
        assert set(source)==set(model.encoder.state_dict()), 'Pretrained tensor keys differ'
        assert all(torch.equal(t.cpu(),source[n]) for n,t in model.encoder.state_dict().items()), 'Pretrained weights modified'
    with torch.inference_mode():
        memory,_=model.encode(images[:2])
        expected_tokens=(args.image_size//14)**2 if args.backbone=='dinov2_small' else ((args.image_size+31)//32)**2
        assert memory.shape==(2,expected_tokens,256)
        prefix=captions[:2,:8].clone()
        changed=prefix.clone()
        changed[:,4:]=7
        first=model.decode(prefix,memory)
        second=model.decode(changed,memory)
        assert torch.allclose(first[:,:4],second[:,:4],atol=1e-5,rtol=1e-5), 'Future-token leakage'
        padded=torch.cat((prefix,torch.zeros(2,3,device='cuda',dtype=torch.long)),1)
        assert torch.allclose(first,model.decode(padded,memory)[:,:8],atol=1e-5,rtol=1e-5), 'Padding changes prefix'
    for p in model.encoder.parameters(): p.requires_grad=False
    model.unfreeze_visual_tail(args.visual_tail_blocks)
    if args.backbone=='dinov2_small':
        for index,block in enumerate(model.encoder.encoder.layer):
            assert all(p.requires_grad == (index>=12-args.visual_tail_blocks) for p in block.parameters())
    model.train()
    model.encoder.eval()
    running={n:t.clone() for n,t in model.encoder.named_buffers() if 'running_' in n}
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    scaler=torch.amp.GradScaler('cuda')
    before=model.output.weight.detach().clone()
    visual_parameter = next(p for p in model.encoder.parameters() if p.requires_grad)
    visual_before=visual_parameter.detach().clone()
    frozen_parameter = next(p for p in model.encoder.parameters() if not p.requires_grad)
    frozen_before=frozen_parameter.detach().clone()
    losses=[]
    torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):
            logits=model(images,captions)
            loss=nn.functional.cross_entropy(logits.flatten(0,1),captions[:,1:].flatten(),ignore_index=PAD,label_smoothing=.05)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        assert model.visual_layers[0].self_attn.in_proj_weight.grad.abs().sum()>0
        for layer in model.decoder.layers:
            assert layer.self_attn.in_proj_weight.grad.abs().sum()>0
            assert layer.multihead_attn.in_proj_weight.grad.abs().sum()>0
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())
    assert not torch.equal(before,model.output.weight)
    assert not torch.equal(visual_before,visual_parameter)
    assert torch.equal(frozen_before,frozen_parameter)
    for name,t in model.encoder.named_buffers():
        if name in running: assert torch.equal(t,running[name]), 'BN statistics changed'
    model.eval()
    with torch.inference_mode():
        memory,projected=model.encode(images[:2])
        beam=search_features(model,memory,projected,max_words=10)
        singles=[search_features(model,memory[i:i+1],projected[i:i+1],max_words=10)[0] for i in range(2)]
        for i,single in enumerate(singles):
            assert beam[i].tolist()==single.tolist(), 'Beam batch isolation'
        full=model.decode(captions[:2,:8],memory)
        state=model.start_state(memory)
        for i in range(8):
            step,state=model.step(captions[:2,i],memory,projected,state)
            assert torch.allclose(step,full[:,i],atol=2e-5,rtol=2e-5), 'Teacher/inference prefix mismatch'
        reference=model(images[:2],captions[:2])
    ck=dict(state_dict=model.state_dict(),config=model.config,vocab=vocab,epoch=0,validation_similarity=None)
    path=out/'smoke_model.pt'
    torch.save(ck,path)
    assert path.stat().st_size<200_000_000
    restored=model_from_checkpoint(torch.load(path,map_location='cpu',weights_only=True)).cuda().eval()
    with torch.inference_mode():
        assert torch.equal(reference,restored(images[:2],captions[:2])), 'Reload mismatch'
    # Legacy construction remains compatible and gives identical output on real images.
    legacy=torch.load(root/'captioning/runs/resnet50_aoa_v1/best_model.pt',map_location='cpu',weights_only=True)
    from train import AttentionCaptioner
    original=AttentionCaptioner(len(legacy['vocab']),pretrained=False,backbone_name='resnet50',attention_type='aoa').eval()
    original.load_state_dict(legacy['state_dict'],strict=True)
    loaded=model_from_checkpoint(legacy).eval()
    with torch.inference_mode():
        assert torch.equal(original.generate(images[:2].cpu()),loaded.generate(images[:2].cpu()))
    old_transformer=torch.load(root/'captioning/runs/resnet50_transformer_v1/best_model.pt',map_location='cpu',weights_only=True)
    old_model=model_from_checkpoint(old_transformer).eval()
    with torch.inference_mode():
        assert torch.isfinite(old_model(images[:2].cpu(),captions[:2].cpu())).all()
    result=dict(checks=['causality','padding','AMP backward','visual and decoder update','BN frozen',
        'beam batch isolation','teacher/inference prefix equivalence','exact reload','legacy compatibility'],
        backbone=args.backbone,visual_tail_blocks=args.visual_tail_blocks,image_size=args.image_size,
        decoder_layers=args.decoder_layers,
        parameters=sum(p.numel() for p in model.parameters()),
        batch_size=24,losses=losses,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        checkpoint_bytes=path.stat().st_size)
    (out/'checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
