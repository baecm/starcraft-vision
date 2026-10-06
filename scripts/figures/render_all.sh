#!/usr/bin/env bash
# scripts/figures/render_all.sh
#
# Regenerates every data figure used in the thesis and the Director-CenterNet
# paper, one `make figure-fg` per figure, with the arguments that produced the
# published versions (taken from the source comments in the paper's main.tex).
# Run from anywhere on a worker with Docker; the figure scripts run inside the
# debugger container. heatmap and graphical_abstract re-run a checkpoint and
# want the GPU.
#
# Output goes to results/figures/qualitative/ (the scripts' default --outdir)
# as .pdf and .png. The documents hold renamed copies of those files. Figures
# drawn at full width are rendered twice: for the paper at 190 mm, calling the
# viewers "observers", and for the thesis (suffix _spectator) at its 155 mm text
# width, calling them "spectators", so that neither document scales the fonts:
#
#   output file (results/figures/qualitative/)      paper (figures/qualitative/)    thesis (figs/)
#   graphical_abstract_4520_10298[_spectator]        graphical_abstract_4520_10298   overview_graphical_abstract.pdf (_spectator)
#   qual0_teaser_4520_10298[_spectator]              qual0_teaser_4520_10298         mrvp_qual_teaser.pdf (_spectator)
#   qual4_supervision_4520_10298[_spectator]         qual4_supervision_4520_10298    mrvp_supervision.pdf (_spectator)
#   qual1_compare_4520_10298[_spectator]             qual1_compare_4520_10298        mrvp_qual_compare.pdf (_spectator)
#   qual2_heatmap_4520_10298[_spectator]             qual2_heatmap_4520_10298        mrvp_qual_heatmap.pdf (_spectator)
#   qual3_trajectory_3613_1395_1445[_spectator]      qual3_trajectory_3613_1395_1445 mrvp_qual_trajectory.pdf (_spectator)
#   qual6_supervision_grid_4[_spectator]             qual6_supervision_grid_4        mrvp_supervision_grid.pdf (_spectator)
#   arch_*.png (export_panels)                       panels inside dcn_architecture  panels inside dcn_architecture.pdf (figs/arch_panels/)
#   single_failure_4664_10331_1725_11970_12097       -                               single_failure.pdf
#   single_failure_4664_10331_1725_11970_12097_observer  single_failure             -
#
# The thesis calls the human viewers "spectators" and the paper "observers",
# hence the second run of every figure that names them. dcn_architecture itself is drawn by hand
# in draw.io around the export_panels images and is not regenerated here.

set -e
cd "$(dirname "$0")/../.."

# Paper: graphical abstract (130 x 50 mm). Thesis: figs/overview_graphical_abstract.pdf.
make figure-fg FIG=graphical_abstract ARGS="--replay 4520 --frame 10298 --director director=dc_full_b16_f1_s456_v6:30"
make figure-fg FIG=graphical_abstract ARGS="--replay 4520 --frame 10298 --director director=dc_full_b16_f1_s456_v6:30 --person spectator --suffix _spectator"
# Paper: qual0_teaser (opening figure). Thesis: figs/mrvp_qual_teaser.pdf.
make figure-fg FIG=teaser ARGS="--replay 4520 --frame 10298"
make figure-fg FIG=teaser ARGS="--replay 4520 --frame 10298 --person spectator --suffix _spectator"
# Paper: qual4_supervision (supervision construction). Thesis: figs/mrvp_supervision.pdf.
make figure-fg FIG=supervision ARGS="--replay 4520 --frame 10298"
make figure-fg FIG_WIDTH_MM=155 FIG=supervision ARGS="--replay 4520 --frame 10298 --person spectator --suffix _spectator"
# Paper: qual1_compare (qualitative figure 1). Thesis: figs/mrvp_qual_compare.pdf.
make figure-fg FIG=compare ARGS="--replay 4520 --frame 10298 --baseline maskrcnn=maskrcnn_win4_vanilla_f1_s456_v6:30@0.5 --director director=dc_full_b16_f1_s456_v6:30"
make figure-fg FIG_WIDTH_MM=155 FIG=compare ARGS="--replay 4520 --frame 10298 --baseline maskrcnn=maskrcnn_win4_vanilla_f1_s456_v6:30@0.5 --director director=dc_full_b16_f1_s456_v6:30 --person spectator --suffix _spectator"
# Paper: qual2_heatmap (qualitative figure 2). Thesis: figs/mrvp_qual_heatmap.pdf.
make figure-fg FIG=heatmap ARGS="--replay 4520 --frame 10298 --director director=dc_full_b16_f1_s456_v6:30"
make figure-fg FIG_WIDTH_MM=155 FIG=heatmap ARGS="--replay 4520 --frame 10298 --director director=dc_full_b16_f1_s456_v6:30 --suffix _spectator"
# Paper: qual3_trajectory (qualitative figure 3). Thesis: figs/mrvp_qual_trajectory.pdf.
make figure-fg FIG=trajectory ARGS="--replay 3613 --start 1395 --end 1445 --baseline maskrcnn=maskrcnn_win4_vanilla_f1_s456_v6:30@0.5 --director director=dc_full_b16_f1_s456_v6:30"
make figure-fg FIG_WIDTH_MM=155 FIG=trajectory ARGS="--replay 3613 --start 1395 --end 1445 --baseline maskrcnn=maskrcnn_win4_vanilla_f1_s456_v6:30@0.5 --director director=dc_full_b16_f1_s456_v6:30 --suffix _spectator"
# Paper: qual6_supervision_grid_4 (supervision on four more frames). Thesis: figs/mrvp_supervision_grid.pdf.
make figure-fg FIG=supervision_grid ARGS="--frames 3613:340 4664:17160 1725:6020 275:44960"
make figure-fg FIG_WIDTH_MM=155 FIG=supervision_grid ARGS="--frames 3613:340 4664:17160 1725:6020 275:44960 --person spectator --suffix _spectator"
# Paper and thesis: the data panels of the hand-drawn architecture figure (dcn_architecture).
make figure-fg FIG=export_panels ARGS="--replay 4520 --frame 10298 --director director=dc_full_b16_f1_s456_v6:30"
# Thesis: figs/single_failure.pdf (single-region baseline chapter; viewers called spectators).
make figure-fg FIG_WIDTH_MM=155 FIG=single_failure ARGS="--replay 4664 --frame 10331 --tie-replay 1725 --start 11970 --end 12097 --pad 10 --baseline maskrcnn=maskrcnn_win4_vanilla_f1_s456_v6:30@0.5 --label Mask_R-CNN --person spectator"
# Paper: figures/qualitative/single_failure (viewers called observers).
make figure-fg FIG=single_failure ARGS="--replay 4664 --frame 10331 --tie-replay 1725 --start 11970 --end 12097 --pad 10 --baseline maskrcnn=maskrcnn_win4_vanilla_f1_s456_v6:30@0.5 --label Mask_R-CNN --person observer --suffix _observer"
