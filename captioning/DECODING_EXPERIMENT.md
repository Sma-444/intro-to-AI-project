# Fixed-weight decoding comparison (2026-11-18)

Model: existing `runs/resnet50_aoa_v1/best_model.pt` (epoch 11, 124,741,979 bytes). No weights updated. All 2,500 validation images and references; original 224 center-crop preprocessing. Test data and root submission artifacts untouched. The original greedy result reproduced exactly in this comparison.

| Policy | Mean validation cosine | Mean words | Repeated-trigram captions |
|---|---:|---:|---:|
| Original greedy | 0.582442880 | 10.3560 | 97 |
| Constrained greedy | 0.582570016 | 10.3952 | 0 |
| Beam width 3, alpha 0 | 0.586542428 | 9.6892 | 0 |
| Beam width 3, alpha 0.6 | **0.586902320** | 9.8844 | 0 |

All policies cap generation at 24 tokens. Constraints suppress PAD/BOS/UNK, prevent EOS at the first step, and block exact repeated token trigrams. These are algorithmic decoding constraints; no reference captions, semantic evaluator, or manual caption edits are used to select a sentence for a particular image. MiniLM only scores the completed validation CSVs. Beam rank is cumulative model log-probability divided by `((5 + length)/6)**alpha`, counting EOS in length. Completed hypotheses retain their score/length in a shared live/completed beam; pruning occurs by the normalized rank at each step. This is an approximate beam search, not exact global search.

The best policy improves by **0.004459441**, without larger saved weights. The alpha 0.6 vs alpha 0 difference is only 0.000359893; do not claim a robust advantage for that exact hyperparameter. These results use the same repeatedly consulted validation split, not independent test evidence. No formal statistical significance is established. Repetition reduction alone barely improves the metric; larger progress toward 0.7 still requires better visual/semantic learning.

## Reproduce

From project root:

```powershell
$env:PYTHONPATH = (Resolve-Path 'captioning/.env/site-packages').Path
python captioning/infer.py --checkpoint captioning/runs/resnet50_aoa_v1/best_model.pt --image-dir Validation --output captioning/runs/decoding_aoa_v1/inference_verified.csv --decode-policy beam3 --length-alpha 0.6
```

`compare_decoding.py` contains the four predefined policies and saves CSVs/results under `runs/decoding_aoa_v1/`; it refuses to overwrite a completed result file. `check_decoding.py` checks a known globally better-than-greedy sequence, batch isolation and EOS score preservation. The comparison checks beam-width-one unconstrained decoding agrees with original greedy text on its first actual validation batch. Source-independent CLI reproduction of all 2,500 descriptions is also verified against the comparison CSV.

Recommended next training experiments (not run here): progressively fine-tune layer3+layer4 with conservative learning rates and frozen BatchNorm statistics; or add training-caption-derived object/action supervision, carefully treating unmentioned objects as noisy negatives. The latter should not use MiniLM vectors, external teacher captions, or reference-dependent inference. A higher-resolution experiment remains untested; the failed 224 letterbox run does not establish that 320 cannot help.
