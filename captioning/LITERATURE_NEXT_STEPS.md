# Literature-informed next experiments (2026-11-18)

## Evidence and limits

This review consulted the original arXiv abstract pages linked below and the installed torchvision weight metadata. It is not a full-text reproduction review. Paper benchmark scores are usually COCO CIDEr/BLEU, not this course's MiniLM similarity. No paper establishes that our 12,000 single-caption training examples can achieve 0.7. All expected benefits below are hypotheses to test using validation only.

Current measured baseline: ResNet-50 ImageNet1K V2 + learned 2048-to-512 projection + additive attention LSTM, full validation similarity 0.577977, complete float32 checkpoint 120,542,243 bytes. Train loss continues falling after the best validation epoch, so capacity alone may not solve the remaining generalization gap.

## Priority 1: Attention on Attention (AoA)

Huang et al., *Attention on Attention for Image Captioning*, ICCV 2019.
https://arxiv.org/abs/1908.06954

The abstract proposes generating both an information vector and attention gate from the query/current context and attended features, then multiplying them to filter the information. Our current additive-attention context is passed directly to the LSTM and output layer. A small AoA-inspired decoder can selectively filter context without replacing the proven visual encoder. Use randomly initialized decoder/gate weights, not the authors' pretrained caption checkpoint. An isolated gate-only addition is an adaptation, not a reproduction of full AoANet (which applies AoA in both encoder and decoder).

Plan: retain ResNet-50, splits, preprocessing, training budget and seed; modify only attention processing. Compare full validation similarity and actual checkpoint size. Hypothesis: lower-risk change than replacing all language modeling with a Transformer; improvement is not guaranteed.

## Priority 2: Modern classification-pretrained encoders

Liu et al., *A ConvNet for the 2020s*, CVPR 2022.
https://arxiv.org/abs/2201.03545

Tan and Le, *EfficientNetV2: Smaller Models and Faster Training*, ICML 2021.
https://arxiv.org/abs/2104.00298

Installed torchvision metadata (classification-model parameters include the classifier, which a caption encoder removes):

| Weight enum | Classification parameters | ImageNet top-1 | Default crop |
|---|---:|---:|---:|
| ResNet50 IMAGENET1K_V2 | 25,557,032 | 80.858% | 224 |
| ConvNeXt_Tiny IMAGENET1K_V1 | 28,589,128 | 82.520% | 224 |
| EfficientNet_V2_S IMAGENET1K_V1 | 21,458,488 | 84.228% | 384 |

Official API references:
- https://docs.pytorch.org/vision/stable/models/generated/convnext_tiny.html
- https://docs.pytorch.org/vision/stable/models/generated/efficientnet_v2_s.html
- https://docs.pytorch.org/vision/stable/models/generated/resnet50.html

Classification accuracy is not caption quality and these recipes/resolutions differ. ConvNeXt-Tiny plus the existing small decoder is expected to remain below 200 MB in float32 (must measure actual checkpoint). EfficientNetV2-S is also promising, but the 384-pixel preset raises compute/memory demand; its published classification accuracy should not be assumed at 224 pixels. First compare ConvNeXt-Tiny at 224 pixels with the existing decoder; separately test any decoder improvement.

## Priority 3: Adaptive visual attention

Lu et al., *Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning*, CVPR 2017.
https://arxiv.org/abs/1612.01887

The decoder can choose between image regions and a visual sentinel representing internal linguistic information. This addresses forcing visual context into every generated word, including function words. Applicable to our recurrent decoder, but test as an alternative to AoA before combining both.

## Higher-complexity alternative: small Transformer with spatial context

Cornia et al., *Meshed-Memory Transformer for Image Captioning*, CVPR 2020.
https://arxiv.org/abs/1912.08226

Uses multi-level visual relationships, learned memory, and meshed connections for decoding. A compact randomly initialized 2-layer Transformer with cross-attention to classification-encoder grid features is a plausible ablation, not a reproduction of M2 Transformer. Full pretrained caption checkpoints or externally generated caption training data are prohibited. With only one caption per training image, overfitting risk and tuning cost may outweigh gains; compare after the smaller changes above.

## Region attention without importing a large detector

Anderson et al., *Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering*, CVPR 2018.
https://arxiv.org/abs/1707.07998

Jiang et al., *In Defense of Grid Features for Visual Question Answering*, CVPR 2020.
https://arxiv.org/abs/2001.03615

The former uses detector-derived regions; the latter reports grid features can remain competitive under comparable pretraining, with simpler/faster systems and applicability to captioning. For this assignment keep classification-pretrained spatial grid features and, if useful, borrow the top-down decoder design instead of adopting detector weights/features trained with external annotations. Neither result proves any arbitrary grid encoder equals a region model.

## Training methods and decoding

Rennie et al., *Self-critical Sequence Training for Image Captioning*, CVPR 2017.
https://arxiv.org/abs/1612.00563

Optimizes sequence metrics with a self-critical reward baseline. A reward computed solely from training captions using CIDEr is conceptually compatible with the data restriction, but one reference per image gives sparse lexical supervision and its target differs from MiniLM. Lower priority. **Do not use MiniLM outputs as a reward, loss, distillation target, or feature**: the course permits it only for validation-based model selection.

Bengio et al., *Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks*, NeurIPS 2015.
https://arxiv.org/abs/1506.03099

Gradually mixes model-generated previous words into teacher-forced training. Could reduce train/inference discrepancy, but is a separate optimization change and not guaranteed to improve validation. Lower priority than encoder/attention ablations.

Implementation-specific low-cost experiments (engineering hypotheses, not claims established by these papers): compare greedy vs beam width 3 with validation-selected length normalization; compare current center crop to aspect-preserving full-image padding or direct resize, using matched train/inference preprocessing for retraining. Cropping can omit objects near borders; padding/stretching introduces its own distribution shift. Do not assume beam search is always better, and do not tune on test predictions.

## Recommended sequence

1. Keep the existing 0.577977 checkpoint as control; inspect validation errors only.
2. ResNet-50 + AoA-inspired attention; keep remaining settings fixed.
3. ConvNeXt-Tiny + original additive attention decoder; keep 224-pixel budget comparable.
4. Combine only components that improve validation; test preprocessing and decoding separately.
5. Recheck promising improvements with another seed if the gain is small; preserve run configs, losses, scores and complete checkpoints.

No new training or test inference was performed as part of this literature review.
