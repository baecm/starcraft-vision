"""Dataset-free check that the heatmap output stride is wired consistently.

Run before launching a stride sweep. It answers the three questions that make a
resolution change unattributable if they are wrong:

1. Does the decoder actually produce the requested grid, and at what cost?
2. Do the cell-unit knobs (NMS kernel, soft-argmax window, border ring) still
   cover the same distance in tiles? They are configured in cells, so a finer
   stride shrinks every one of them unless they are re-derived.
3. Does L_hcm keep its magnitude? Its negative term sums over cells while
   num_pos is fixed at one per sample, so a 4x finer grid would make it 16x
   larger and, under one global gradient-norm clip, starve every other
   objective. That is the same failure the squared-L2 L_smooth produced.

It also asserts that the *configured* values on the model stay unscaled, since
_write_run_provenance records them and inference rebuilds from that record - a
model that scaled them in place would apply the factor again on every reload.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/workspace/src")

import torch

from utils.torch_compat import disable_inductor

disable_inductor()

from losses.director_losses import human_consensus_match_loss  # noqa: E402
from models.backbones.director_centernet import (  # noqa: E402
    FPN_STRIDE,
    NMS_RADIUS_TILES,
    DirectorCenterNet,
)
import config  # noqa: E402

MAP = 128


def main() -> int:
    x = torch.randn(2, 36, MAP, MAP)
    print(f"input {tuple(x.shape)}  FPN level stride {FPN_STRIDE}\n")

    for stride in (4, 2, 1):
        for head_conv in (config.DIRECTOR_HEAD_CONV, 256):
            model = DirectorCenterNet(
                in_channels=36, down_ratio=stride, head_conv=head_conv
            ).eval()
            with torch.no_grad():
                feat = model._extract_features(x)
                pred_hm, _logits, _off, _wh = model._predict_heads(feat)

            params = sum(p.numel() for p in model.parameters()) / 1e6
            act = pred_hm.new_zeros(()).element_size() * feat.numel() / 2 / 1e6

            # Distances the derived knobs actually cover, in tiles.
            nms_tiles = (model._nms_kernel - 1) // 2 * stride
            soft_tiles = model._soft_center_cells * stride
            border_tiles = model._peak_border_cells * stride

            pos = torch.zeros_like(pred_hm)
            pos[:, :, 5, 5] = 1.0
            l_hcm = human_consensus_match_loss(
                pred_hm=pred_hm,
                target_hm_top1=torch.zeros_like(pred_hm),
                pos_mask=pos,
                neg_weight=model._hcm_neg_weight,
            ).item()

            print(
                f"stride={stride} head_conv={head_conv:3d} | "
                f"grid={tuple(pred_hm.shape[-2:])} params={params:5.1f}M "
                f"feat_act={act:5.1f}MB/sample | "
                f"nms={model._nms_kernel}x{model._nms_kernel}({nms_tiles:.0f}t) "
                f"soft_r={model._soft_center_cells}({soft_tiles:.0f}t) "
                f"border={model._peak_border_cells}({border_tiles:.0f}t) | "
                f"neg_w={model._hcm_neg_weight:.4f} L_hcm@init={l_hcm:.3f}"
            )

            assert pred_hm.shape[-1] == MAP // stride, "decoder produced the wrong grid"
            assert model.soft_center_radius == config.DIRECTOR_SOFT_CENTER_RADIUS, (
                "configured soft_center_radius was scaled in place; provenance "
                "would make inference scale it a second time"
            )
            assert model.peak_border_margin == config.DIRECTOR_PEAK_BORDER_MARGIN, (
                "configured peak_border_margin was scaled in place"
            )
            assert abs(nms_tiles - NMS_RADIUS_TILES) < stride, (
                f"NMS radius drifted to {nms_tiles} tiles at stride {stride}"
            )
        print()

    try:
        DirectorCenterNet(in_channels=36, down_ratio=3)
    except ValueError as exc:
        print(f"stride=3 correctly rejected: {exc}")
    else:
        print("FAIL: stride=3 was accepted but is not a divisor of the FPN stride")
        return 1

    print("\nall consistency checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
