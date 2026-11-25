# ResNet-50 architecture experiment

## Results (2026-11-18)

Training completed normally after 11 epochs (early stopping); best checkpoint is epoch 7. Full validation re-evaluation after reloading `best_model.pt` gives **0.577977**, compared with **0.528876** for the original ResNet-18 model, an absolute improvement of **0.049101**. The training-time score was 0.577920; small differences across runs are possible with CUDA mixed precision and cuDNN benchmark enabled.

The complete checkpoint is **120,542,243 bytes**, within the strict 200,000,000-byte limit. All 2,500 reloaded validation predictions have matching unique IDs and nonempty descriptions. This experiment improves the validation metric but does not reach 0.7, establish test performance, or establish statistical significance across random seeds. Root `model/final_model.pt` and `test_predictions.csv` still refer to the older ResNet-18 run; this experiment saved its own best model and did not run test inference.

Result artifacts: `runs/resnet50_attn_lstm_v1/best_model.pt`, `verification.json`, `training_history.csv`, `validation_predictions_best.csv`, `validation_predictions_reloaded.csv`, `config.json`, `environment.json`, and `console.log`.

## Hypothesis and scope

ResNet-18's visual features may limit object and attribute recognition. This experiment replaces it with torchvision's ImageNet classification pretrained ResNet-50 (`IMAGENET1K_V2`). A learned 2048-to-512 linear projection maps each 7x7 spatial location into the existing attention-LSTM interface. The decoder is initialized from scratch, including the projection. No pretrained language decoder, vision-language model, external caption dataset, or text-embedding supervision is used.

This comparison changes the encoder architecture AND its classification-pretraining recipe; it is not an ablation isolating depth alone. A single seed is an initial comparison rather than a guarantee of statistically robust improvement.

The baseline is the saved ResNet-18 model, validation similarity 0.528876. Validation uses all 2,500 supplied validation pairs with the course MiniLM evaluator (max text length 64, attention-mask mean pooling, L2 normalization, mean pairwise cosine similarity). MiniLM outputs are used only to select checkpoints. Test images/predictions are not used for this experiment or model selection.

## Matched settings

- Training: all 12,000 training pairs; vocabulary built only from training captions (frequency >=2); seed 2205.
- Same 224-pixel training crop and mild color jitter as baseline; evaluation resize 256 + center crop 224.
- Same attention-LSTM embedding=256, hidden=512, dropout=0.3; teacher forcing, token cross-entropy with 0.05 label smoothing, gradient clipping 1.0.
- Decoder/projection AdamW learning rate 3e-4, layer4 learning rate 2e-5, weight decay 1e-4.
- Freeze encoder for 2 epochs, then fine-tune layer4; learning rates decay by 0.8 after each fine-tuning epoch.
- Batch size 24; mixed precision; maximum 16 epochs; stop after 4 epochs without a new best validation cosine score.
- Greedy decoding, maximum 24 generated tokens, matches the baseline.

## Local reproduction

From the project root, using the existing isolated package installation:

```powershell
$env:PYTHONPATH = (Resolve-Path 'captioning/.env/site-packages').Path
$env:TORCH_HOME = (Resolve-Path 'captioning/.weights').Path
python -u captioning/train.py --backbone resnet50 --output-dir captioning/runs/resnet50_attn_lstm_v1 --batch-size 24 --eval-batch-size 48 --workers 2 --cpu-threads 8 --epochs 16
```

Use a NEW output directory for a rerun. The training script refuses to overwrite a previous run's history/checkpoints. `config.json`, `training_history.csv`, `console.log` (captured by the original launch command), last/best checkpoints, and current/best validation CSVs are in the experiment directory.

The ResNet-50 weight download was verified against the official filename SHA256 prefix `11ad3fa6` (full SHA256: `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`). This file is an ImageNet image-classification encoder resource, not a caption model.

After training, re-load and independently evaluate the saved best model:

```powershell
python captioning/evaluate_saved.py --checkpoint captioning/runs/resnet50_attn_lstm_v1/best_model.pt
```

The checker verifies all 2,500 validation IDs, nonempty predictions, strict weight loading, and the actual checkpoint file size <200,000,000 bytes. It writes `verification.json` and `validation_predictions_reloaded.csv` into the experiment directory. Original baseline checkpoint and final test prediction files remain available separately.
