# archive/legacy/

Code moved out of `src/`, `scripts/` and `conf/` on 2026-10-07 because no
current experiment uses it. It is kept for reference, under the same relative
paths it had; it is **not** on any import path and is not expected to run as
it stands. The last commit where it ran in place is `52a4313`
(`git checkout 52a4313` to use it).

| Moved | What it was |
|---|---|
| `src/models/backbones/rtdetr.py`, `conf/architecture/rtdetr.yaml` | RT-DETR observer (smoke-tested only) |
| `src/models/backbones/deformable_detr.py`, `conf/architecture/deformable_detr.yaml`, `src/models/utils/box_ops.py` | Deformable DETR observer and its box helpers |
| `src/models/backbones/deformable_video_detr.py`, `spatiotemporal_encoder.py` | Video DETR with a CVAE rater query (`ProbabilisticVideoDETR`) |
| `src/models/plugins/probabilistic_query.py`, `cvae_query.py`, `conf/plugins/probabilistic_query/` | Latent-query plugins of the DETR variants |
| `src/losses/unified_losses.py` | Energy / hysteresis losses of the DETR variants |
| `scripts/run_benchmark.py`, `src/tools/run_benchmark.py`, `src/evaluate_benchmark.py` | End-to-end benchmark (`make benchmark`, `evaluate.py --benchmark`). It fell back to an untrained model when a checkpoint failed to load, so treat any number it produced as suspect |
| `src/preprocessing/pair.py` | Paired-frame preprocessing; had no entry point |

Deleted outright (dead code, no caller; recoverable from git history):
`src/detection/{engine,coco_eval,coco_utils,transforms}.py`,
`src/metrics/multi_region_eval.py`, `src/models/plugins/kbrs/eval.py`,
`src/estimate.py` (a re-export of `evaluate.py`; `make estimate` already ran
`evaluate.py`), and the unused helpers of `src/metrics/custom_evaluator.py`
(`kernel_scores`, `eval_run`, the window/RLE helpers).

`commands.sh` still lists commands for these models; it is a log of past
commands, not something to run.
