# H100 runbook (2026-10-12 .. 10-25)

Commands for the dedicated H100 window, in queue order. Every run is
launched from the tag **`h100-2026-10`** (commit `4c3aefd`), so each run's
W&B tags carry `git:4c3aefd80`. Do not train on worker07.

```bash
git fetch --tags && git checkout h100-2026-10
```

Before the first run on the machine, warm the page cache once (the first
epoch otherwise waits on the CIFS share; see the training-throughput note):

```bash
find /mnt/nas/baecm/starcraft-vision/data -type f -print0 | xargs -0 -P 32 -n 64 cat > /dev/null
```

Queue order: **A1 → A2** (thesis, first week), **A3** alongside when the
Director runs fit next to a Mask R-CNN run, then **fold 2/3** (ToG only; drop
fold 3 first if time runs short). Run the KBRS and vanilla arms of one seed on
the same GPU type.

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

## A3 — Director-CenterNet with the unmodified CornerNet focal loss

**Not implemented yet.** The configuration switch has to be added and
CPU-verified before 10/12; this section gets its commands then.

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

- Check the W&B tags show `git:4c3aefd80` and the intended `kbrs:` /
  `targets:` / `batch:` values.
- Copy nothing by hand: checkpoints, predictions and results are on the NAS
  under the names above.
