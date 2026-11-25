# Local Captioning Experiment

Current highest measured candidate: **DINOv2-small + 4-layer Transformer, 336 input, seed2206**, independently verified validation cosine **0.690574467**. Selected epoch12; checkpoint **111,731,457 bytes** at `runs/dinov2_transformer_336_decoder4_seed2206/best_model.pt`. All 2500 training-selected, reloaded and CLI-generated captions match. Four layers improve over two in both fixed seeds; the two-seed mean gain is **0.003851772**, but this is not proof of stable generalization. The fixed same-trajectory averaging trial scored **0.689012110** and was rejected. See [SEED_AND_WEIGHT_AVERAGING.md](SEED_AND_WEIGHT_AVERAGING.md).

Previous single-seed candidate is **DINOv2-small + 4-layer Transformer at 336 input**, independently reloaded validation cosine **0.689380348**, up **0.003118873** over the 2-layer control. Selected epoch13 of 16; complete checkpoint **111,731,457 bytes** at `runs/dinov2_transformer_336_decoder4_v1/best_model.pt`. All 2,500 training-selected, independently reloaded and CLI-generated captions match exactly. The paired bootstrap difference interval includes zero; this single-seed gain is not established as stable. See [DINO_DECODER_DEPTH_EXPERIMENT.md](DINO_DECODER_DEPTH_EXPERIMENT.md).

The retained **2-layer Transformer at 336 input** scores **0.686261475**, up **0.010294139** over 224 input. Selected epoch15 of 16; checkpoint **103,290,761 bytes** at `runs/dinov2_transformer_336_v1/best_model.pt`. See [DINO_RESOLUTION_EXPERIMENT.md](DINO_RESOLUTION_EXPERIMENT.md) for the resolution comparison and cuDNN backend reproducibility findings.

The earlier 224-input **DINOv2-small + compact Transformer** scored **0.675967336** with beam3/alpha0.6, improving over ResNet50 + Transformer (**0.595902145**) by **0.080065191**. Training stopped at epoch15 and selected epoch11. Its checkpoint remains at `runs/dinov2_transformer_v1/best_model.pt`. See [DINO_TRANSFORMER_EXPERIMENT.md](DINO_TRANSFORMER_EXPERIMENT.md). Fine-tuning four DINO blocks instead of two scored **0.674344242**; see [GENERALIZATION_AND_TAIL4.md](GENERALIZATION_AND_TAIL4.md). Root submission artifacts remain the older model and have not been replaced.

The previous ResNet50 + Transformer candidate and its checkpoint at `runs/resnet50_transformer_v1/best_model.pt` remain available. See [MODERN_CAPTIONING_AND_TRANSFORMER.md](MODERN_CAPTIONING_AND_TRANSFORMER.md) for that experiment and verified industrial references.

The previous best AoA checkpoint with constrained beam search (width 3, length alpha 0.6) scored **0.586902**, compared with greedy **0.582443**. Use `infer.py --decode-policy beam3 --length-alpha 0.6` with `runs/resnet50_aoa_v1/best_model.pt`. See [DECODING_EXPERIMENT.md](DECODING_EXPERIMENT.md) for validation-only results and verified reproduction. Legacy checkpoints still default to greedy; new checkpoints may specify their decoding policy.

The new ResNet50 + compact Transformer trial uses a randomly initialized text decoder, fixed beam3 checkpoint selection, and separate resumable training state. See [MODERN_CAPTIONING_AND_TRANSFORMER.md](MODERN_CAPTIONING_AND_TRANSFORMER.md) for architecture, verified industrial references, checks, and run commands.

Full-image preprocessing comparison: 224-pixel letterbox scored **0.576513**, below the crop-based AoA model (**0.582443**). See [FULL_IMAGE_EXPERIMENT.md](FULL_IMAGE_EXPERIMENT.md). Preprocessing and input size are now stored in new checkpoints and read by both evaluation and inference; old checkpoints retain the original crop behavior.

Latest comparison: ResNet-50 + AoA-inspired gate reached **0.582443** validation cosine (124.74 MB), while ConvNeXt-Tiny + original decoder reached **0.570801** (134.91 MB). See [ARCHITECTURE_COMPARISON.md](ARCHITECTURE_COMPARISON.md) for settings, results, and reproduction. These are separate experiments; the original root `model/final_model.pt` and `test_predictions.csv` have not been replaced. Inference supports all saved variants via their checkpoint configuration.

For the stronger ResNet-50 experiment (validation similarity 0.577977; 120.54 MB complete checkpoint), see [UPGRADE_EXPERIMENT.md](UPGRADE_EXPERIMENT.md). Its results and files are separate from the original ResNet-18 run described below. `infer.py` supports both checkpoint architectures.

This experiment uses an ImageNet-pretrained ResNet-18 image encoder and a randomly initialized LSTM decoder with additive spatial attention. The text decoder is trained only on `Training_Labels.csv`. The frozen course-provided MiniLM model is used only to score validation predictions, as required by the project README.

## Method choice

`Show and Tell` established an image encoder plus recurrent language decoder trained with caption likelihood. `Show, Attend and Tell` adds spatial attention so each generated word can use a different image region. Bottom-Up and Top-Down Attention reports stronger region-level features, but its Faster R-CNN detector adds complexity and model size that are unnecessary for this small, constrained project. A Transformer decoder is a valid alternative, but training all language-model parameters from scratch on 12,000 single-caption examples raises data and tuning demands. The selected ResNet-18 plus attention-LSTM keeps the decoder learnable from this dataset, uses an explicitly permitted ImageNet image-classification encoder, and stays well below the 200 MB checkpoint cap.

The encoder is initially frozen while the decoder learns; then only ResNet layer 4 is unfrozen at a lower learning rate. Training uses teacher-forced token cross-entropy with light label smoothing, mild crop/color augmentation, and gradient clipping. The best checkpoint is selected by the course's frozen MiniLM mean cosine similarity on all 2,500 validation images. Test images are not used during training or model selection.

## Environment

The current local run was tested with Python 3.14.3, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, and an RTX 5060 Laptop GPU. Install packages into an isolated project package directory and set `PYTHONPATH` when running commands:

```powershell
$site = "captioning\.env\site-packages"
python -m pip install --target $site -r captioning\requirements.txt --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
$env:PYTHONPATH = (Resolve-Path $site).Path
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The first run downloads the allowed ImageNet ResNet-18 classification weights. The course text embedding model is loaded locally from `Text_Embedding_Model/`; no external text model is downloaded or used for generation.

## Train

From `AIAA2205Project/`:

```powershell
python captioning\train.py --batch-size 24 --workers 4
```

The script writes checkpoints under `captioning/runs/resnet18_attn_lstm/` and validation predictions to `captioning/validation_predictions.csv`. It evaluates every epoch and stops after four epochs without a new best validation score. Override the data/output directories and training settings with the command-line options shown by `python captioning\train.py --help`.

The final selected checkpoint is `best_model.pt`. It includes the ResNet-18 ImageNet weights, decoder weights, and vocabulary; verify its uncompressed size is below 200,000,000 bytes before submission.

## Inference

Generate predictions from a split with the selected checkpoint:

```powershell
$env:PYTHONPATH = (Resolve-Path "captioning\.env\site-packages").Path
python captioning\infer.py --checkpoint captioning\runs\resnet18_attn_lstm\best_model.pt --image-dir Test --output captioning\test_predictions.csv
```

The output has exactly the required `image_id,description` columns and uses filename stems as IDs. Only run this on `Test/` after model selection is complete.

## References

- Vinyals et al., “Show and Tell: A Neural Image Caption Generator,” CVPR 2015. https://arxiv.org/abs/1411.4555
- Xu et al., “Show, Attend and Tell: Neural Image Caption Generation with Visual Attention,” ICML 2015. https://arxiv.org/abs/1502.03044
- He et al., “Deep Residual Learning for Image Recognition,” CVPR 2016. https://arxiv.org/abs/1512.03385
- Anderson et al., “Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering,” CVPR 2018. https://arxiv.org/abs/1707.07998
- Vaswani et al., “Attention Is All You Need,” NeurIPS 2017. https://arxiv.org/abs/1706.03762

