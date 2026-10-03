#!/usr/bin/env python3
"""
qualitative_figures.py
======================

Draws the three qualitative figures of the Director-CenterNet paper
(Section "Qualitative Analysis") from real frames:

  compare     (1) one frame, three panels: the U observers and their ranked
                  modes / the baseline's top-K boxes / Director-CenterNet's
                  primary + auxiliary regions
  heatmap     (2) the same frame as four panels: coverage a_t / smoothed
                  coverage with ranked modes and support / predicted heatmap
                  Y_hat_t / decoded regions over the minimap
  trajectory  (3) the primary-region centre over a window of consecutive
                  frames for both methods, with the Top-1 / Top-2 mode tracks
                  and the tie frames (support margin 0) shaded

and a fourth subcommand that picks the frames:

  select      reads the frames_<name>.csv that mode_disagreement.py wrote for
              both methods and lists candidate frames for (1)/(2) and
              candidate windows for (3), together with the numbers the caption
              needs to say how the example was chosen.

A single frame cannot show that one method is better, and the paper's own
tables say Mask R-CNN leads on OC3@0.5 and that VD does not separate the
methods. `select` therefore ranks frames by how close they sit to a chosen
quantile of the per-frame difference, not by how flattering they are, and
prints where the chosen example falls so the caption can state it.

Predictions are read the way mode_disagreement.py reads them: COCO files under
{pred-root}/{model}/model_{epoch:03d}[_th<x>]/{replay}.rep/{label-method}.json,
with every box forced to the fixed viewport size anchored at its top-left.
The heatmap is not saved by inference, so `heatmap` re-runs the Director
checkpoint on that one frame and checks the decoded regions against the saved
predictions before drawing anything.

Usage
-----
Run inside the debugger container (it mounts scripts/, models/, predictions/,
results/ and data/):

  # 0. per-frame CSVs, if not already there
  make mode-disagreement-fg ARGS="--replays 275 1725 3613 4520 4664 \
      --model maskrcnn=<maskrcnn_run> --model director=dc_full_b16_f1_s456_v6 \
      --epoch 30 --outdir /workspace/results/mode_disagreement/fold1"

  # 1. candidates
  python3 /workspace/scripts/qualitative_figures.py select \
      --baseline-csv /workspace/results/mode_disagreement/fold1/frames_maskrcnn.csv \
      --director-csv /workspace/results/mode_disagreement/fold1/frames_director.csv

  # 2. figures
  python3 /workspace/scripts/qualitative_figures.py compare \
      --replay 1725 --frame 4312 \
      --baseline maskrcnn=<maskrcnn_run>:30 --director director=dc_full_b16_f1_s456_v6:30
  python3 /workspace/scripts/qualitative_figures.py heatmap \
      --replay 1725 --frame 4312 --director director=dc_full_b16_f1_s456_v6:30
  python3 /workspace/scripts/qualitative_figures.py trajectory \
      --replay 1725 --start 4200 --end 4500 \
      --baseline maskrcnn=<maskrcnn_run>:30 --director director=dc_full_b16_f1_s456_v6:30

or through make: make qualitative-figures-fg ARGS="compare --replay ...".

Figures go to --outdir (default /workspace/results/figures/qualitative) as
both .pdf and .png, named after the subcommand, replay and frame.

Each subcommand now lives in its own script under scripts/figures/ (the
selection subcommands in figures/select_frames.py), which can be run directly:
make figure-fg FIG=compare ARGS="--replay ...". This file only dispatches to
them under the old names and arguments; scripts/figures/render_all.sh has the
commands behind the published figures.
"""

from __future__ import annotations

import argparse
import os
import sys

_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from figures import (
    architecture,
    compare,
    export_panels,
    graphical_abstract,
    heatmap,
    select_frames,
    single_failure,
    supervision,
    supervision_grid,
    teaser,
    trajectory,
)
from figures.common import common_parser


def _figure(name, module):
    return name, module.HELP, module.add_args, module.run


_SELECT = {c[0]: c for c in select_frames.COMMANDS}

# (subcommand, help, add_args, run), in the order the single script defined them
SUBCOMMANDS = [
    _SELECT["select"],
    _figure("teaser", teaser),
    _figure("export-panels", export_panels),
    _figure("architecture", architecture),
    _figure("graphical-abstract", graphical_abstract),
    _figure("supervision-grid", supervision_grid),
    _SELECT["select-supervision"],
    _figure("supervision", supervision),
    _figure("compare", compare),
    _figure("heatmap", heatmap),
    _figure("trajectory", trajectory),
    _SELECT["select-single"],
    _figure("single-failure", single_failure),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    common = common_parser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, help_, add_args, run in SUBCOMMANDS:
        p = sub.add_parser(name, parents=[common], help=help_)
        add_args(p)
        p.set_defaults(func=run)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
