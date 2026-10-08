# H100 runbook (2026-10-12 .. 10-25)

Commands for the dedicated H100 window, in queue order. Every run is
launched from the tag **`h100-2026-10`**, so each run's W&B `git:` tag must equal
`git rev-parse --short=9 h100-2026-10`. Do not train on worker07.

```bash
git fetch --tags --force && git checkout h100-2026-10
```

Before the first run of a fold on the machine, warm the page cache: read the
fold's training replays once, so the first epoch does not wait on the CIFS
share (it was 2 h 18 min against 1 h 28 min for later epochs in a DGX
Mask R-CNN run). Run it from the repo root with the run's fold; it reads only
the ten training replays, not the test replays, the KBRS cache or the audit
directories under `data/`. It changes nothing, can be stopped at any time,
and training can start while it runs. On a machine whose RAM cannot hold the
ten replays it helps little; skip it there.

```bash
FOLD=1; R=$(sed -n 's/^train_replays: *\[\(.*\)\]/\1/p' conf/dataset/fold$FOLD.yaml | tr -d ' ' | tr ',' ' '); (cd /mnt/nas/baecm/starcraft-vision/data/input/dst && for r in $R; do find $r.rep -type f -print0; done | xargs -0 -P 32 -n 64 cat > /dev/null)
```

Check that the machine has the ImageNet ResNet-50 weights (in
`.torch_cache`, or network access to download them). Mask R-CNN stops if
they are missing, but Director-CenterNet only logs `Failed to load ImageNet
weights` and trains from an untrained backbone, which would make A3 a
different experiment. The first Director log must say `[Model]
imagenet_backbone: True`.

What to read in a training log (`logs/<container>.log`):

- start-up: `[Env]` (GPU, torch, CUDA), `[Run] W&B tags`, `[Data] detector
  targets`, the `[Model]` lines (the settings the model actually uses: input
  size, KBRS weight and parameters, Director loss weights and
  `hcm_negative_target`), `[Provenance] commit` (must not say DIRTY), and
  `[Train]` (iterations per epoch, batch, loader workers);
- every 5 minutes: `[Epoch e] i/n  eta ...  loss ... lr ... grad ... s/it
  (data ...)  mem ...`; a data time close to s/it means the run waits on the
  NAS;
- every epoch: `[Epoch e/30] <time>  loss ...  lr ...  run ETA ...`;
- any skipped batch at once: `[NaN] epoch e iter i: ...`.

The same goes to W&B: epoch charts (`Loss/*`, `Train/lr`,
`Time/s_per_iter`, `Time/data_wait_frac`, `System/max_mem_GB`) against
`epoch`, within-epoch ones (`Iter/*`) against `iter`, and the model's
settings under `effective` in the run config.

Queue order: **A1 → A2** (thesis, first week), **A3** alongside when the
Director runs fit next to a Mask R-CNN run, then **fold 2/3** (ToG only; drop
fold 3 first if time runs short). Run the KBRS and vanilla arms of one seed on
the same GPU type.

## Run list

Tick a run when its checkpoint `model_030.pth` exists and its W&B tags are
right. Hours are rough: Mask R-CNN is compute-bound (about 12 h on a 5090,
measure the first H100 run and correct), Director-CenterNet input-bound
(about 6 h) and able to share the GPU with a Mask R-CNN run. 21 Mask R-CNN
runs at ~10 h fill about 210 of the window's ~330 hours.

| # | Group | Run (`id_string`) | Model | Seed | For | Done |
|---|---|---|---|---|---|---|
| 1 | A1 | `maskrcnn_win4_kbrs_relu_nogate_f1_s1001_b16_v6` | Mask R-CNN + KBRS | 1001 | thesis RQ3, ToG | [ ] |
| 2 | A1 | `maskrcnn_win4_vanilla_f1_s1001_b16_v6` | Mask R-CNN | 1001 | thesis RQ3, ToG | [ ] |
| 3 | A1 | `maskrcnn_win4_kbrs_relu_nogate_f1_s2002_b16_v6` | Mask R-CNN + KBRS | 2002 | thesis RQ3, ToG | [ ] |
| 4 | A1 | `maskrcnn_win4_vanilla_f1_s2002_b16_v6` | Mask R-CNN | 2002 | thesis RQ3, ToG | [ ] |
| 5 | A1 | `maskrcnn_win4_kbrs_relu_nogate_f1_s3003_b16_v6` | Mask R-CNN + KBRS | 3003 | thesis RQ3, ToG | [ ] |
| 6 | A1 | `maskrcnn_win4_vanilla_f1_s3003_b16_v6` | Mask R-CNN | 3003 | thesis RQ3, ToG | [ ] |
| 7 | A2 | `maskrcnn_win4_modes_soft_f1_s123_v6` | Mask R-CNN, mode targets (soft) | 123 | thesis ch. 8–9 | [ ] |
| 8 | A2 | `maskrcnn_win4_modes_soft_f1_s456_v6` | Mask R-CNN, mode targets (soft) | 456 | thesis ch. 8–9 | [ ] |
| 9 | A2 | `maskrcnn_win4_modes_soft_f1_s789_v6` | Mask R-CNN, mode targets (soft) | 789 | thesis ch. 8–9 | [ ] |
| 10 | A3 | `dc_hcm_only_cornernet_b16_f1_s123_v6` | Director, hcm_only, CornerNet weight | 123 | thesis ch. 9 | [ ] |
| 11 | A3 | `dc_hcm_only_cornernet_b16_f1_s456_v6` | Director, hcm_only, CornerNet weight | 456 | thesis ch. 9 | [ ] |
| 12 | A3 | `dc_hcm_only_cornernet_b16_f1_s789_v6` | Director, hcm_only, CornerNet weight | 789 | thesis ch. 9 | [ ] |
| 13 | fold 2 | `maskrcnn_win4_kbrs_relu_nogate_f2_s123_b16_v6` | Mask R-CNN + KBRS | 123 | ToG | [ ] |
| 14 | fold 2 | `maskrcnn_win4_vanilla_f2_s123_b16_v6` | Mask R-CNN | 123 | ToG | [ ] |
| 15 | fold 2 | `maskrcnn_win4_kbrs_relu_nogate_f2_s456_b16_v6` | Mask R-CNN + KBRS | 456 | ToG | [ ] |
| 16 | fold 2 | `maskrcnn_win4_vanilla_f2_s456_b16_v6` | Mask R-CNN | 456 | ToG | [ ] |
| 17 | fold 2 | `maskrcnn_win4_kbrs_relu_nogate_f2_s789_b16_v6` | Mask R-CNN + KBRS | 789 | ToG | [ ] |
| 18 | fold 2 | `maskrcnn_win4_vanilla_f2_s789_b16_v6` | Mask R-CNN | 789 | ToG | [ ] |
| 19 | fold 3 | `maskrcnn_win4_kbrs_relu_nogate_f3_s123_b16_v6` | Mask R-CNN + KBRS | 123 | ToG | [ ] |
| 20 | fold 3 | `maskrcnn_win4_vanilla_f3_s123_b16_v6` | Mask R-CNN | 123 | ToG | [ ] |
| 21 | fold 3 | `maskrcnn_win4_kbrs_relu_nogate_f3_s456_b16_v6` | Mask R-CNN + KBRS | 456 | ToG | [ ] |
| 22 | fold 3 | `maskrcnn_win4_vanilla_f3_s456_b16_v6` | Mask R-CNN | 456 | ToG | [ ] |
| 23 | fold 3 | `maskrcnn_win4_kbrs_relu_nogate_f3_s789_b16_v6` | Mask R-CNN + KBRS | 789 | ToG | [ ] |
| 24 | fold 3 | `maskrcnn_win4_vanilla_f3_s789_b16_v6` | Mask R-CNN | 789 | ToG | [ ] |
| 25 | A2 (optional) | `maskrcnn_win4_modes_hard_f1_s123_v6` | Mask R-CNN, mode targets (hard) | 123 | thesis ch. 8–9 | [ ] |

Each run is followed by its inference and analysis (sections below), which
need no training time but do need the GPU for Mask R-CNN inference.

### After the thesis (not implemented; not part of this window)

Retraining for the ESWA paper, to be designed after the defense. Each is a
new objective, so each needs code, a CPU check and its own runbook entry first.

| Idea | Aimed at |
|---|---|
| (A) penalize heatmap peaks where no spectator looks | the ~72 % of off auxiliary regions that cover no viewport |
| (B) a decoder-aware false-peak loss | peaks the decoder turns into off regions |
| (C) a floor on minority-mode amplitude | low-support modes (less promising: off regions do not score low) |
| (D) a mode-count head | the region count itself |

---

## A1 — KBRS controlled comparison, fold 1, seeds 1001 / 2002 / 3003

Same configuration as the existing three seeds
(`maskrcnn_win4_{kbrs_relu_nogate,vanilla}_f1_s{123,456,789}_b16_v6`, trained
at `416c4ef`; the training path is unchanged up to `h100-2026-10`). Those runs
were launched with exactly these overrides (Hydra `overrides.yaml` on the NAS,
2026-10-02):

| arm | overrides |
|---|---|
| KBRS | `plugins/kbrs=enabled plugins.kbrs.kbrs_params.gate_channels=[] plugins.kbrs.kbrs_params.mixture_nonneg=relu batch_size=16` |
| vanilla | `batch_size=16` |

Everything else is the default (fold 1, window 4, 30 epochs, cosine, lr 0.005,
KBRS weight 0.25, Mask R-CNN input 640x640).

For each S in 1001, 2002, 3003:

```bash
make train ARGS="plugins/kbrs=enabled plugins.kbrs.kbrs_params.gate_channels=[] plugins.kbrs.kbrs_params.mixture_nonneg=relu batch_size=16 seed=S id_string=maskrcnn_win4_kbrs_relu_nogate_f1_sS_b16_v6"
```
```bash
make train ARGS="batch_size=16 seed=S id_string=maskrcnn_win4_vanilla_f1_sS_b16_v6"
```

Inference at the four thresholds of the existing seeds:

```bash
for m in kbrs_relu_nogate vanilla; do for th in 0.5 0.7 0.8 0.9; do make inference-fg ARGS="--model-name maskrcnn_win4_${m}_f1_sS_b16_v6 --model-number 30 --replays 275 1725 3613 4520 4664 --window-size 4 --score-threshold $th --architecture maskrcnn --cuda"; done; done
```

Analysis, laid out like `results/mode_disagreement/kbrs_rerun_s{123,456,789}`:

```bash
make mode-disagreement-fg ARGS="--replays 275 1725 3613 4520 4664 --model kbrs_th05=maskrcnn_win4_kbrs_relu_nogate_f1_sS_b16_v6@0.5 --model kbrs_th07=maskrcnn_win4_kbrs_relu_nogate_f1_sS_b16_v6@0.7 --model kbrs_th08=maskrcnn_win4_kbrs_relu_nogate_f1_sS_b16_v6@0.8 --model kbrs_th09=maskrcnn_win4_kbrs_relu_nogate_f1_sS_b16_v6@0.9 --model vanilla_th05=maskrcnn_win4_vanilla_f1_sS_b16_v6@0.5 --model vanilla_th07=maskrcnn_win4_vanilla_f1_sS_b16_v6@0.7 --model vanilla_th08=maskrcnn_win4_vanilla_f1_sS_b16_v6@0.8 --model vanilla_th09=maskrcnn_win4_vanilla_f1_sS_b16_v6@0.9 --outdir /workspace/results/mode_disagreement/kbrs_rerun_sS"
```

Report: the IR difference KBRS − vanilla per seed, over all six seeds; state
that seeds 1001–3003 were trained on the H100 and 123–789 on worker08/09.

---

## A2 — Mask R-CNN trained on ranked-mode boxes, fold 1, seeds 123 / 456 / 789

Compared against the v6 Mask R-CNN baseline
`maskrcnn_win4_vanilla_f1_s{123,456,789}_v6` (batch 32, default config,
trained at `bed81e3`; the vanilla training path is unchanged since). So A2
keeps batch 32 and the same seeds; the only change is `mode_targets`.

For each S in 123, 456, 789:

```bash
make train ARGS="mode_targets=soft seed=S id_string=maskrcnn_win4_modes_soft_f1_sS_v6"
```

With `mode_targets=soft` the box score is trained toward support / U, so a
mode held by one or two spectators scores about 0.2–0.4 and a 0.5 filter
drops it. Infer over a lower threshold range as well:

```bash
for th in 0.1 0.2 0.3 0.5 0.7 0.9; do make inference-fg ARGS="--model-name maskrcnn_win4_modes_soft_f1_sS_v6 --model-number 30 --replays 275 1725 3613 4520 4664 --window-size 4 --score-threshold $th --architecture maskrcnn --cuda"; done
```

Analysis: `mode_disagreement` (as above, methods `soft_th01` … `soft_th09`,
outdir `.../modes_soft_sS`), then `count_validity.py` and `off_diagnosis.py`
against the baseline at its matched-budget threshold:

```bash
make analysis SCRIPT=count_validity ARGS="--model soft_th02=maskrcnn_win4_modes_soft_f1_sS_v6@0.2 --model soft_th05=maskrcnn_win4_modes_soft_f1_sS_v6@0.5 --model mrcnn_th09=maskrcnn_win4_vanilla_f1_sS_v6@0.9"
```

Question it answers: are the baseline's redundant (dup) and empty (off)
boxes, and its negative beta_n, a product of training on five overlapping
viewports per frame?

If H100 time is left over: one `mode_targets=hard` seed separates the effect
of mode boxes from that of the support weighting.

---

## A3 — Director-CenterNet with the unmodified CornerNet focal loss, fold 1, seeds 123 / 456 / 789

The thesis says beta_n is already positive with L_hcm alone, and that L_hcm
departs from CornerNet. In the `hcm_only` configuration (L_rmc, L_rep,
L_smooth off) the auxiliary ignore mask is already off, so the one remaining
departure is the negative weight: `(1 - Y_all)^beta`, which protects the
auxiliary modes, where CornerNet uses the primary's own Gaussian
`(1 - Y1)^beta`. `architecture.hcm_negative_target=primary` switches to the
latter (and would also drop the ignore mask in configurations with L_rmc).

Compared against `dc_hcm_only_b16_f1_s{123,456,789}_v6` (trained at
`aae0d85`/`bed81e3`; the Director training path is unchanged since). Those
were launched with `make run` (train, then inference at the architecture's
score threshold 0.1 into `model_030/`), so A3 uses `make run` too.

For each S in 123, 456, 789:

```bash
make run ARGS="architecture=director_centernet batch_size=16 seed=S architecture.loss_weights.lambda_rmc=0 architecture.loss_weights.lambda_rep=0 architecture.loss_weights.lambda_sm=0 architecture.hcm_negative_target=primary id_string=dc_hcm_only_cornernet_b16_f1_sS_v6"
```

W&B tags must show `loss:hcm_only` and `hcm_neg:primary`; the log must show
`[Model] hcm_negative_target: primary` and `[Model] imagenet_backbone: True`.

Analysis next to the existing ablation (`results/mode_disagreement/v6_f1_sS`):

```bash
make mode-disagreement-fg ARGS="--replays 275 1725 3613 4520 4664 --model hcm_only=dc_hcm_only_b16_f1_sS_v6 --model hcm_only_cornernet=dc_hcm_only_cornernet_b16_f1_sS_v6 --outdir /workspace/results/mode_disagreement/a3_cornernet_sS"
```
```bash
make analysis SCRIPT=count_validity ARGS="--model hcm_only=dc_hcm_only_b16_f1_sS_v6 --model hcm_only_cornernet=dc_hcm_only_cornernet_b16_f1_sS_v6"
```

Question it answers: does beta_n stay positive without the joint-target
protection, i.e. is the sign a property of the heatmap formulation or of the
modified loss? Director runs are input-bound, so they can share the H100 with
a Mask R-CNN run.

---

## Fold 2 / 3 — KBRS controlled comparison for the ToG revision

Same two arms as A1, seeds 123 / 456 / 789, `dataset=fold2` or `dataset=fold3`.

| fold | test replays |
|---|---|
| 2 | 1559 1628 2351 6219 11251 |
| 3 | 36 212 438 522 1660 |

For each fold F in 2, 3 and seed S:

```bash
make train ARGS="dataset=foldF plugins/kbrs=enabled plugins.kbrs.kbrs_params.gate_channels=[] plugins.kbrs.kbrs_params.mixture_nonneg=relu batch_size=16 seed=S id_string=maskrcnn_win4_kbrs_relu_nogate_fF_sS_b16_v6"
```
```bash
make train ARGS="dataset=foldF batch_size=16 seed=S id_string=maskrcnn_win4_vanilla_fF_sS_b16_v6"
```

Inference and analysis as in A1, with that fold's test replays and outdir
`.../kbrs_rerun_fF_sS`.

---

## After each run

- Check the W&B tags show the tag's commit (`git:`) and the intended `kbrs:` /
  `targets:` / `batch:` values.
- Copy nothing by hand: checkpoints, predictions and results are on the NAS
  under the names above.
