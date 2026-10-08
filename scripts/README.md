# scripts/

Analysis, figure and sanity-check scripts. None of them trains a model; the
analysis scripts read stored predictions and ground truth only and need no GPU.

Run any analysis script with

```bash
make analysis SCRIPT=<name without .py> ARGS="..."
```

Each script's docstring has its full usage. Raw outputs of the 2026-10
analyses are on the NAS in `results/analysis_2026-10/`, whose README maps each
one to its thesis and ESWA section.

## Analysis scripts

| Script | Question it answers | Reads | Reported in |
|---|---|---|---|
| `mode_disagreement.py` | Per-frame mode attribution, IR, OC_k, flip rate and M-CTI for each method. Writes `frames_<method>.csv`, which the next three scripts read | GT + predictions | thesis `tab:mrvp:primary`, `tab:mrvp:multiregion`, `tab:att:nmodes`, `tab:att:margin`, `tab:att:progression`; ESWA |
| `budget_allocation.py` | How each method spends its region count across the number of modes (under-emission, deficit). **Its OC is over answered frames only** | `frames_*.csv` | thesis `tab:mrvp:budget` |
| `split_by_second_mode.py` | Count response and coverage split by whether the second mode is held by one spectator or by two or more | `frames_*.csv` | thesis `tab:mrvp:split` |
| `switch_preparation.py` | When the primary region cuts, was the destination already shown as an auxiliary region? | GT + predictions (+ `frames_*.csv` for the per-n breakdown) | thesis `sec:disc:smooth` |
| `count_validity.py` | Is beta_n made of useful regions? Splits it into new / dup / off, against chance placements | GT + predictions | thesis `tab:mrvp:countsplit`, ESWA |
| `off_diagnosis.py` | What the auxiliary regions that cover no mode are, and whether a score filter removes them | GT + predictions | thesis `sec:mrvp:budget` (off regions), ESWA 5.6 |
| `aux_on_modes.py` | Do auxiliary regions (top k only) land on new modes, in rank order? | GT + predictions | thesis `tab:mrvp:auxland` |
| `hybrid_regions.py` | A proposal detector's primary plus a heatmap detector's auxiliaries, combined after the fact | GT + two models' predictions | thesis `tab:mrvp:hybrid` |
| `mode_sensitivity.py` | How the mode structure depends on the extraction parameters (sigma, theta, D) | GT only | thesis `tab:att:sensitivity`; ESWA |
| `kbrs_cue_scores.py` | KBRS cue scores (density, centeredness, mixture) at the observers' and each model's primary viewport, raw and per-frame normalized, from the input-frame KBRS cache | KBRS cache + GT + predictions | thesis `tab:kbrs:scores`, ToG |
| `pool_size_convergence.py` | Has the mode structure leveled off at U = 5 spectators? Subsamples U = 2..5 | GT only | thesis `sec:att:convergence`; ESWA Limitations |
| `reeval_missing_frames.py` | Single-region IR under each evaluator fix, to reproduce and correct earlier numbers | predictions | ToG revision notes, thesis ch. 7 |

`analysis_common.py` holds what these scripts share: the `NAME=MODEL[:EPOCH][@THRESHOLD]`
model spec, the CLI options, loading one replay's ground truth and one model's
regions, the new / dup / off classification and the beta_n slope. Read its
docstring for the box conventions.

### Conventions every analysis script follows

- **Frames are pooled** over all replays given (not averaged per replay).
- **A declined frame scores 0** (coverage-penalized), except the OC column of
  `budget_allocation.py`, which is over answered frames.
- **beta_n** is fitted over frames with 1–5 modes, a declined frame counting
  zero regions.
- **Predicted regions** are forced to the observers' viewport size at their
  stored top-left corner.
- **Mode extraction** uses sigma 4, theta 0.35, D 12, at most 5 modes
  (`src/config.py`), unless a script's options change them.

## Other scripts

| Script | What it does |
|---|---|
| `figures/` | One script per published figure; `figures/render_all.sh` has the commands. Run with `make figure-fg FIG=<name>` |
| `qualitative_figures.py` | Older entry point that dispatches to `figures/`; kept for `make qualitative-figures-fg` |
| `run_registry.py` | Which trained runs are comparable to the current code (`make run-registry`) |
| `probe_director.py`, `probe_resolution.py`, `kbrs_smoke.py` | Model sanity checks; they produce no reported numbers |
| `run_benchmark.py` | Legacy end-to-end benchmark (moving to `archive/`) |
