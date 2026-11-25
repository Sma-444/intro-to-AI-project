# Targeting validation similarity above 0.7

Update after the first proposed experiment: full-image letterbox224 was trained and independently evaluated, obtaining **0.576513**, below the crop-based AoA control **0.582443**. It early-stopped at epoch 13 and selected epoch 9. The predeclared +0.002 gate for a follow-up 320-pixel run was not met, so 320 training has not run. Details and limitations are in `FULL_IMAGE_EXPERIMENT.md`. The remaining experiments below are still untested; this result does not establish that higher-resolution full-frame processing cannot help.

The target is aspirational, not guaranteed. Current best full-validation score is 0.582443; reaching 0.7 requires +0.117557 absolute improvement. Diagnostic results were computed from the existing AoA validation predictions using the frozen course evaluator. No model fitting or test-set access was performed for this analysis.

## Measured bottlenecks

- 540/2500 examples score below 0.4 (mean 0.269310). The inspected lowest-scoring reference/prediction pairs include entirely different subjects and actions. This is a text-pair observation, not an image-grounded correctness audit: inspecting validation images is needed to distinguish model errors, ambiguous scenes and noisy references.
- 824/2500 score at least 0.7 (mean 0.828481). This is not an accuracy statistic or evidence that the overall target is attainable.
- 97/2500 predictions have repeated trigrams (mean 0.508587). Even hypothetically improving all of those to perfect similarity would add only about 0.0191 to the overall mean. Repetition fixes alone cannot reach 0.7.
- Reference and prediction mean word counts are 10.3724 and 10.3560; length mismatch alone does not explain the score. Counting visible words cannot determine whether a decoder hit its token cap because special tokens are omitted by decoding.
- Resize-short-side-to-256 then center-crop-224 retains approximately 55.61% of original image area on average across validation image dimensions, ignoring integer rounding. This is a geometric calculation, not proof that the discarded area contains the relevant subject. Training random near-square crops can also remove described content while keeping the full original caption.
- Best AoA validation score peaks at epoch 11 while training token loss keeps declining afterward. Merely extending the same training is not a strong proposal.

Sources: `runs/resnet50_aoa_v1/diagnostics.json`, `validation_diagnostics.csv`, `training_history.csv`.

## Prioritized controlled experiments

1. **Full-image input**: retain the proven ResNet-50 + AoA architecture and compare current crop preprocessing against aspect-preserving letterbox padding to 224 pixels. Apply the matching preprocessing during training/inference, with restrained color augmentation. This isolates crop information loss without changing architecture or resolution. Padding can also hurt because of pretraining distribution shift; measure rather than assume.
2. **Resolution**: if full-image input helps, compare 224 versus 320 pixels (7x7 versus 10x10 feature grid for this encoder). Reduce microbatch and use gradient accumulation as needed to keep the effective batch comparable. Resolution does not increase saved parameter count but raises activation memory and compute; run a real-batch test first.
3. **Visual adaptation**: compare fine-tuning only layer4 versus progressively unfreezing layer3+layer4 at separate low learning rates. Test keeping BatchNorm running statistics frozen, since small caption batches can destabilize pretrained statistics. Keep decoder regularization and early stopping; deeper unfreezing can overfit.
4. **Explicit training-only concept supervision**: build a fixed noun/action/color vocabulary from training captions only. Add a lightweight multilabel concept head and BCE auxiliary objective to the image encoder, optionally feed predicted concept probabilities into the decoder. Compare against the same configuration without the auxiliary head. Captions omit visible concepts, so absent words are noisy negatives; weight/regularize the auxiliary objective and do not assume it solves grounding. Do not use MiniLM outputs or external language/vision APIs to create targets.
5. **Decoding ablation**: on an unchanged checkpoint, compare greedy vs beam size 3 with length normalization, special-token suppression and optional repeated-trigram constraints. Select on validation only. This can address search/repetition errors, but cannot repair missed visual subjects; better likelihood may not improve MiniLM similarity.
6. **Only then consider a new decoder family**: a compact Transformer with learned spatial positions and cross-attention, trained from scratch on provided training captions, is a separate experiment if substantial grounding errors remain. No claim that it must outperform LSTM with this dataset size.

Use fixed comparisons and preserve checkpoints and run configurations. Avoid claiming additive gains across changes without measuring their combination. Small improvements should be checked across seeds; repeated validation selection can overfit validation. Do not use test images, references or predictions for decisions. Frozen MiniLM is permitted only for validation scoring/selection, never as training loss, reward, feature or distillation target. All complete checkpoints must stay below 200,000,000 bytes.

Original first planned run: **ResNet-50 + AoA, full-image letterbox at 224, with other training settings held fixed**. This experiment is now complete (see update above). Progressive visual unfreezing and decoding comparisons remain candidates; there is no empirical justification yet for claiming they will achieve 0.7.
