# AoA and ConvNeXt-Tiny comparison

## Completed results (2026-11-18)

Both new runs finished normally after 15 epochs via early stopping. Both selected epoch 11. Each checkpoint was then loaded independently and evaluated against all 2,500 validation references with the frozen course MiniLM model.

| Model | Reloaded validation cosine | Difference vs ResNet-50 additive | Full checkpoint bytes |
|---|---:|---:|---:|
| Original ResNet-18 + additive attention | 0.528876 | -0.049101 | 66,779,443 |
| ResNet-50 + additive attention (control) | 0.577977 | 0 | 120,542,243 |
| ResNet-50 + AoA-inspired gate | **0.582443** | **+0.004466** | 124,741,979 |
| ConvNeXt-Tiny + additive attention | 0.570801 | -0.007176 | 134,909,179 |

The AoA-inspired model is the best measured candidate, but its advantage is small and measured at one seed; do not claim statistical significance or 0.7-level performance. ConvNeXt-Tiny underperformed the matched ResNet-50 control under this recipe, so combining it with AoA is not supported by these results. This does not establish that ConvNeXt cannot improve with its own normalization, resolution, or fine-tuning settings.

Both verification scripts passed strict weight loading, exact validation ID matching, uniqueness, nonempty descriptions, and <200,000,000-byte checkpoint size. Small differences between training-time scores (AoA 0.582326; ConvNeXt 0.570818) and reload scores are consistent with CUDA mixed precision/cuDNN differences already observed in earlier runs. No test-set inference or editing of the existing root submission artifacts took place.

Recommended next controlled experiments: validation-only decoding comparison or full-image preprocessing with ResNet-50. A repeated-seed check is useful before attributing the small AoA improvement to architecture alone.

These two experiments follow the literature review in `LITERATURE_NEXT_STEPS.md`. Each uses all 12,000 training pairs and all 2,500 validation pairs. No test images or predictions enter training or selection.

## Variants

1. **ResNet-50 + AoA-inspired context gate**: additive spatial attention first produces context `c`; concatenate the previous LSTM hidden state `h` and `c`, then compute `Linear_info([h,c]) * sigmoid(Linear_gate([h,c]))`. This filtered 512-dimensional context replaces raw context in the LSTM input and output classifier. The rest of the ResNet-50 model stays the same. This is a decoder gate inspired by Huang et al. (ICCV 2019), **not a reproduction of the complete AoANet**: https://arxiv.org/abs/1908.06954
2. **ConvNeXt-Tiny + original additive attention LSTM**: replace ResNet-50 with torchvision `ConvNeXt_Tiny_Weights.IMAGENET1K_V1`, discard the classification head, project the final 768-dimensional spatial features to 512 dimensions, and keep the original decoder. The final ConvNeXt feature stage is unfrozen after two frozen epochs. Uses only ImageNet classification-pretrained visual weights: https://arxiv.org/abs/2201.03545

All decoder/projection/gate weights are initialized from scratch. No external captions, pretrained text decoder, VLM, or teacher captions are used. The course MiniLM is used only for validation metric evaluation and checkpoint selection, never for parameter fitting or rewards.

## Matched protocol

Seed 2205, batch 24, 224-pixel train crops, resize256/center224 validation, vocabulary frequency >=2, embedding 256, LSTM hidden size 512, dropout 0.3, greedy decoding capped at 24 tokens. Decoder/gate/projection AdamW LR=3e-4, final encoder stage LR=2e-5 after 2 frozen epochs, weight decay 1e-4, per-epoch fine-tuning LR decay 0.8, label smoothing 0.05, gradient clipping 1.0, mixed precision. Maximum 16 epochs and 4-epoch validation patience, identical to the preceding ResNet-50 run. One seed per variant; these runs are not a statistical significance study.

## Reproduce

From the project root:

```powershell
$env:PYTHONPATH = (Resolve-Path 'captioning/.env/site-packages').Path
$env:TORCH_HOME = (Resolve-Path 'captioning/.weights').Path
python -u captioning/run_comparison.py
```

Existing experiment directories are protected against overwrite. For a new run, invoke `train.py` with a fresh `--output-dir`, choosing either `--backbone resnet50 --attention aoa` or `--backbone convnext_tiny --attention additive`. See `run_comparison.py` for all arguments. After training, `evaluate_saved.py` re-loads the saved best checkpoint with no pretrained-weight download and validates all IDs, nonempty outputs, and the strict 200 MB size limit.

Results, per-epoch history, original training source snapshot, configurations, console logs, and verification logs are saved separately in:
- `runs/resnet50_aoa_v1/`
- `runs/convnext_tiny_attn_v1/`

Aggregate re-evaluation results are written to `runs/architecture_comparison.json`. Earlier checkpoints, the root submission model and existing test CSV are preserved. No test inference is included in this comparison.

## Preflight validation

Each variant passed 3 AMP optimizer steps with a real batch of 24 training image/caption pairs, finite loss and gradient norm, parameter-update checks, autoregressive decoding, and exact float32 logits comparison after a state_dict save/load roundtrip. AoA gate gradient was nonzero. State-only smoke checkpoints were about 124.7 MB (AoA) and 134.9 MB (ConvNeXt); actual complete checkpoint sizes are recorded in post-training verification. The previous ResNet-50 checkpoint still loads strictly after the code changes.

ConvNeXt source weight SHA256: `983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d` (matches the official filename prefix).
