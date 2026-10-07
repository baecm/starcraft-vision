# StarCraft Vision: automatic observing from spectator viewports

Code for learning where a StarCraft: Brood War broadcast camera should look,
from the viewports of several human spectators of the same replay. It backs
the PhD thesis, the Director-CenterNet paper (ESWA) and the saliency-prior
(KBRS) paper (IEEE ToG). Everything runs in Docker on the Linux workers.

## Models

| `architecture=` | Model | Used for |
|---|---|---|
| `maskrcnn` | Mask R-CNN on the stacked frame window, one box per spectator viewport | the single-region observer and the proposal-detector baseline; with `plugins/kbrs=enabled` the KBRS saliency prior, with `roci=true` ROCI targets, with `mode_targets=hard/soft` ranked-mode boxes |
| `director_centernet` | Director-CenterNet: a heatmap of ranked attention modes, decoded into a primary region and auxiliary regions | the multi-region observer |
| `centernet` | plain CenterNet | the heatmap baseline |

Mask R-CNN resizes its 128x128 tile input to 640x640 (`models/factory.py`;
the `resize` in `conf/architecture/maskrcnn.yaml` is not read). RT-DETR,
Deformable DETR and the video DETR variants are in `archive/legacy/`.

## Pipeline

```text
replays ──preprocess_input──► data/input/dst/<replay>.rep/<frame>.npy        game state per frame
spectators ─preprocess_label─► data/label/dst/<replay>.rep/<method>.json    viewports (COCO boxes)
            ──train──────────► models/<run>/model_NNN.pth, run_provenance.json
            ──inference──────► predictions/<run>/model_NNN[_th<x>]/<replay>.rep/<method>.json
            ──evaluate───────► results/<run>_eNN.csv                        IR, Intersection@, multi-region
            ──scripts/───────► the analyses behind the thesis tables (scripts/README.md)
```

## Repository

```text
conf/                 Hydra config: config.yaml, architecture/, dataset/ (folds), plugins/kbrs/
infra/                Dockerfile, docker-compose.yml, entrypoint.sh (command word -> script)
Makefile              every entry point (make train / inference / evaluate / analysis / test ...)
src/
  train.py            training (Hydra); the module docstring lists its steps in order
  inference.py        checkpoint -> prediction files
  evaluate.py         prediction files -> replay-level metrics
  pipeline.py         train then inference (make run)
  cli.py, config.py   inference CLI; constants (channels, viewport size, mode extraction)
  dataset/            starcraft_windows.py (the training dataset), loader.py, label and mode caches
  models/             factory.py (build_model), backbones/, plugins/kbrs/, plugins/density_peak.py
  losses/             director_losses.py (Director-CenterNet objectives)
  detection/          engine_safe.py (the training loop, with NaN handling)
  metrics/            custom_evaluator.py (IR), evaluator.py (multi-region), modes.py (ranked modes)
  preprocessing/      replay and label preprocessing
  tools/              KBRS cache / lookup / profiling, label prechecks
scripts/              analysis and figure scripts (scripts/README.md)
tests/                unit tests (make test)
archive/legacy/       code no current experiment uses (archive/legacy/README.md)
commands.sh           a log of past commands, not a script to run
```

## Conventions

- **Every accuracy number is frame-pooled and coverage-penalized**: frames
  are pooled over the replays of a fold, and a frame the model does not
  answer scores 0.
- **Boxes** are `[x, y, w, h]` in tiles with `(x, y)` the top-left corner;
  mode centers are `(row, col)`. A predicted region is forced to the
  spectators' viewport size (12 x 20 tiles) at its stored top-left corner.
- **Ranked modes** come from `metrics.modes.extract_modes` with sigma 4,
  theta 0.35, D 12 and at most 5 modes (`src/config.py`).
- **A run records its commit and host** in `run_provenance.json` (training)
  and `inference_provenance.json` (inference); launch through the Makefile,
  which passes them in.
- Do not train on worker07 (suspected bad RAM).

## Common commands

```bash
# preprocessing
make preprocess_input ARGS="--replays 36 212 438 --include-components worker ground air building vision"
make preprocess_label ARGS="--replays 36 212 438 --method all_correct"

# training (Hydra overrides)
make train ARGS="architecture=director_centernet seed=123 id_string=dc_full_b16_f1_s123_v6 batch_size=16"
make train ARGS="seed=123 batch_size=32 id_string=maskrcnn_win4_vanilla_f1_s123_v6"
make train ARGS="plugins/kbrs=enabled seed=123 batch_size=16 id_string=..."
make train ARGS="mode_targets=soft seed=123 id_string=maskrcnn_win4_modes_soft_f1_s123_v6"

# inference and evaluation
make inference ARGS="--model-name <run> --model-number 30 --replays 275 1725 3613 4520 4664 --window-size 4 --score-threshold 0.5 --cuda"
make evaluate ARGS="--mode model --model-name <run> --epoch 30 --replays 275 1725 3613 4520 4664"

# analyses (scripts/README.md) and tests
make analysis SCRIPT=count_validity ARGS="--model dcn=dc_full_b16_f1_s123_v6"
make test
```
