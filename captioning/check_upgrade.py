"""Check backward compatibility and real-caption CUDA training before a new run."""
from pathlib import Path
import argparse
import io
import time
import torch
from train import (AttentionCaptioner, CaptionDataset, build_vocab, collate_train,
                   get_transforms, read_labels, BOS, EOS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', default='resnet50')
    parser.add_argument('--attention', default='additive')
    parser.add_argument('--preprocessing', default='resize256_centercrop224')
    parser.add_argument('--image-size', type=int, default=224)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    torch.manual_seed(2205)
    torch.set_num_threads(8)
    assert torch.cuda.is_available()
    rows = read_labels(root / "Training_Labels.csv")
    vocab = build_vocab(rows)
    transform, _ = get_transforms(args.preprocessing, args.image_size)
    dataset = CaptionDataset(rows[:24], root / "Training", vocab, transform, 34)
    images, captions, _ = collate_train([dataset[i] for i in range(24)])
    assert captions[:, 0].eq(BOS).all() and captions.eq(EOS).sum() == 24
    images, captions = images.cuda(), captions.cuda()
    model = AttentionCaptioner(len(vocab), backbone_name=args.backbone, attention_type=args.attention).cuda()
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    for p in model.encoder[-1].parameters():
        p.requires_grad_(True)
    model.train()
    model.encoder.eval()
    model.encoder[-1].train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    torch.cuda.reset_peak_memory_stats()
    before = model.visual_projection.weight.detach().clone()
    start = time.perf_counter()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(images, captions)
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), captions[:, 1:].reshape(-1), ignore_index=0)
        assert torch.isfinite(loss)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        assert torch.isfinite(norm)
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    assert not torch.equal(before, model.visual_projection.weight)
    if args.attention == 'aoa':
        assert model.aoa_gate.weight.grad is not None
        assert model.aoa_gate.weight.grad.norm().item() > 0
    print(f"PASS: real captions batch=24, 3 AMP optimizer steps, loss={loss.item():.4f}, "
          f"seconds={time.perf_counter()-start:.2f}, peak_allocated_GB={torch.cuda.max_memory_allocated()/1e9:.3f}", flush=True)
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        output = model.generate(images[:2], max_words=8)
    assert output.shape[0] == 2
    print(f"PASS: autoregressive decoding shape={tuple(output.shape)}", flush=True)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    assert buffer.tell() < 200_000_000
    print(f'Checkpoint state bytes: {buffer.tell()}', flush=True)
    buffer.seek(0)
    restored = AttentionCaptioner(len(vocab), backbone_name=args.backbone,
        attention_type=args.attention, pretrained=False).cuda().eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    with torch.inference_mode():
        assert torch.allclose(model(images[:2], captions[:2]), restored(images[:2], captions[:2]))
    print('PASS: strict checkpoint roundtrip logits', flush=True)


if __name__ == "__main__":
    main()
