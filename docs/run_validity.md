# Which runs are valid, and from which commit

A checkpoint whose code has since changed is indistinguishable, in a directory
listing, from one that is still current. An earlier round lost three ablations
to that: they were launched from stale checkouts on other machines and silently
reproduced the previous round, caught only afterwards by checksumming
predictions. This file records the boundaries so the mistake is not repeated by
reading a run's name and assuming.

The **live** table is generated, not maintained here:

```
make run-registry SINCE=5dfede3 ARGS="--out /workspace/results/run_registry.md"
```

It reads the `run_provenance.json` that `train.py` writes next to every
checkpoint. Regenerate it rather than trusting the snapshot below.

## Boundary commits

A run is only comparable to another run if no commit between them changed what
the model computes. These are the commits that did.

| Commit | Date | What it invalidates |
| --- | --- | --- |
| `5dfede3` | 2026-09-23 | **The current boundary.** `StepLR(step_size=3, gamma=0.1)` stepped once per epoch, so a 30-epoch run sat at 5e-6 or below for 24 of its epochs and was effectively over after epoch six. Replaced by cosine over `max_epoch`. Built in `run_training` for **every** architecture, so Mask R-CNN and the KBRS variants are under the same fault, not only Director. The same commit fixed a randomly initialised decoder that could not converge in six epochs. |
| `544359a`, `5fdf73f`, `11b95d7`, `ed86508` | 2026-09-18/19 | Render, head initialisation, peak extraction and loss domains. Superseded by `5dfede3` for practical purposes: anything they invalidate, it invalidates too. |

Everything committed after `5dfede3` has touched metrics, scripts or
preprocessing only, so runs from any of those commits are comparable to each
other.

## What a stale run looks like in the results

The ablation lattice was flat to within seed noise before `5dfede3`, and that
was read as a fact about the objectives. It was the schedule: every optional
loss acts late or weakly, and the learning rate was already dead by the time
they did. `L_smooth` in particular had never been trained at the weight it was
reported with. A pre-boundary number is not noisy, it is measuring something
else.

## Snapshot, 2026-09-24

Post-boundary runs only. `make run-registry` lists the other 117.

| Run | Commit | down_ratio | Predictions | Note |
| --- | --- | --- | --- | --- |
| `dc_full_b16_f1_s456_v6` | `96e69b88` | 4 | 030, 030_th0.5 | the reference Director run |
| `dc_full_b16_s2_f1_s456_v6` | `c9fdda24` | 2 | 030 | stride-2 resolution experiment |
| `dc_full_b16_f1_s789_v6` | `c9fdda24` | 4 | - | still training on 2026-09-24 |

Nothing between `c9fdda24` and `96e69b88` touched model code (only
`src/metrics/modes.py`, `scripts/mode_disagreement.py` and the paper), so the
first two are a clean stride comparison.

Knob signature shared by all three, from `run_provenance.json`:
`k_max=3 conf_threshold=0.1 render_sigma=4.0 u_observers=5
soft_center_radius=2 peak_border_margin=1 trainable_layers=5 head_conv=64
dense_positives=False`.

### Runs that are not usable at all

- `dc_no_rep_f1_s456_v3`, `dc_no_smooth_f1_s456_v3` were trained from a dirty
  working tree and cannot be reproduced from any commit.
- 105 of the 120 stored runs predate `run_provenance.json` and report UNKNOWN.
  Their commit cannot be recovered; they are dated before the boundary by
  filesystem timestamp, which is evidence but not provenance.

## What still has to be retrained

Every baseline the paper compares against is pre-boundary. Under a
fold1 x 3-seed protocol that is seven runs:

- Mask R-CNN vanilla, fold1, seeds 123/456/789
- Director full, fold1, seed 123 (456 done, 789 training)
- `hcm_only`, fold1, seeds 123/456/789 - the paper's "CenterNet (single)" row.
  `dc_hcm_only_f1_s456_v4` exists but is pre-boundary.

Under the full three-fold, three-seed protocol it is 25. The KBRS variants are
out of the paper and are not in either count.

A run can be moved between machines: `checkpoint_every: 5` leaves a checkpoint
every five epochs, and `resume=/workspace/models/<run>/model_020.pth` restores
the optimizer and the schedule position, not only the weights.
