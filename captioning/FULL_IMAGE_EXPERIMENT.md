# Full-image input experiment

## Completed result (2026-11-18)

The 224-pixel run completed normally and early-stopped after epoch 13; epoch 9 was selected. Re-loading the selected checkpoint and re-evaluating all 2,500 validation pairs produced **0.576513**, versus the existing cropped-input AoA control **0.582443**, an absolute difference of **-0.005930**. Thus this run did not improve the model and did not reach the 0.7 objective.

The full checkpoint is **124,741,979 bytes**. Verification passed strict state loading, all 2,500 matching unique IDs, nonempty descriptions, and the <200,000,000-byte limit. Training-time best was 0.575931; the independent reload was 0.576513. AMP/cuDNN implementation differences can change greedy token choices and cause small score variation; these runs are not bitwise-deterministic and no statistical significance is claimed.

The run did not meet the predeclared +0.002 improvement gate, so no 320-pixel training was performed. This decision conserves the experiment budget, **not evidence that 320 pixels cannot help**. The 224 ablation preserves full field of view but also reduces object scale for elongated images, adds padding, and removes random-crop augmentation. Its failure does not prove crop information loss is harmless, nor identify which competing effect caused the reduction.

The existing best candidate remains `runs/resnet50_aoa_v1/best_model.pt` (0.582443). The new checkpoint, logs, source snapshots, configuration, history and predictions remain under `runs/resnet50_aoa_letterbox224_v1/` for review. `runs/full_image_comparison.json` contains the measured comparison. No root submission model or test prediction file was changed.

Suggested next independent experiment: retain the stronger crop-based AoA recipe and compare visual adaptation (progressively unfreeze layer3+layer4 with conservative learning rates and frozen BatchNorm statistics) or decoding. A 320-pixel full-image run would be a separate hypothesis about object scale, rather than a demonstrated improvement from this run.

## Predeclared protocol

Control: saved ResNet-50 + AoA cropped-input model, full-validation reloaded cosine 0.5824428797. New variant: same architecture and ImageNet weights, with the decoder initialized from scratch, seed 2205; replace random training crops and validation center crops with aspect-preserving whole-image resize and centered RGB mean-color padding at 224 pixels. The same mild color jitter is applied to the image before padding, so synthetic padding remains neutral. The ablation measures both preservation of the field of view and removal of random crop augmentation; it does not isolate one of these effects alone.

Matched settings: 12,000 training pairs, 2,500 validation pairs, training-only vocabulary, batch 24, maximum 16 epochs, 2 frozen-encoder epochs, then layer4 fine-tuning, decoder LR 3e-4, encoder LR 2e-5, fine-tuning LR decay 0.8, dropout 0.3, label smoothing 0.05, AdamW weight decay 1e-4, gradient clipping 1.0, AMP, 4 stale validation epochs for early stopping. Frozen MiniLM is only used to evaluate generated validation descriptions. No test access or test-based decisions.

Before seeing this run's score, the follow-up threshold was set: only if reloaded 224 full-image similarity exceeds the cropped-input control by more than 0.002, run a second otherwise identical 320-pixel experiment (after a 24-image resource/optimizer-step check). This is an experiment-selection heuristic, not a significance test. Both variants train from the classification encoder and random decoder; the 320 run does not warm-start from the 224 caption checkpoint.

## Implementation and verification

- `Letterbox` uses bilinear resize preserving aspect ratio with centered padding; all original image content is included (downsampling still loses fine detail).
- Checkpoint metadata stores preprocessing and image size; independent evaluation/inference reads these fields. Old checkpoints fall back to the unchanged 224 center-crop pipeline.
- `check_preprocessing.py` verifies colored edge preservation and aspect ratio for wide/tall/square images at 224/320, deterministic transforms, and exact compatibility of the legacy evaluation transform.
- A real batch of 24 training pairs passed 3 AMP updates, finite gradient checks, AoA gate gradient check, decoding, and a state-dictionary roundtrip. The 224 trial used ~0.54 GB peak allocated tensor memory.
- `run_full_image.py` captures logs, source snapshots, environment versions, configurations, history, best/last checkpoints, and independent full-validation verification. Existing run directories cannot be overwritten.

## Reproduce

From project root with the existing isolated dependencies:

```powershell
$env:PYTHONPATH = (Resolve-Path 'captioning/.env/site-packages').Path
python -u captioning/run_full_image.py
```

Run outputs: `captioning/runs/resnet50_aoa_letterbox224_v1/`, and conditionally `captioning/runs/resnet50_aoa_letterbox320_v1/`. Aggregate results go to `captioning/runs/full_image_comparison.json`. Root submission model/test CSV are preserved while experiments are evaluated on validation only.
