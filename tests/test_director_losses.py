# tests/test_director_losses.py
import pytest
import torch
import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import config
from losses.director_losses import (
    render_gaussian_heatmap_targets,
    human_consensus_match_loss,
    ranked_mode_coverage_loss,
    spatial_repulsion_loss,
    trajectory_smoothness_loss,
    pairwise_box_iou,
)


def test_render_targets_and_masking():
    # 1 batch sample with 2 modes.
    # Centers are multiples of stride so they land exactly on sampled grid
    # points; otherwise the discrete grid never samples the continuous peak
    # (e.g. row=30 with stride=4 is 2 tiles off the nearest grid row, which
    # alone scales the peak down by exp(-2^2 / 2*2^2) ~= 0.61).
    # Top-1 mode at (row=32, col=40) with support=5 -> peak 5/5 = 1.0
    # Top-2 mode at (row=80, col=88) with support=3 -> peak 3/5 = 0.6
    modes_list = [
        {
            "centers": np.array([[32.0, 40.0], [80.0, 88.0]]),
            "support": np.array([5, 3]),
            "n_observers": 5,
        }
    ]
    feat_h, feat_w = 32, 32
    stride = 4
    device = torch.device("cpu")

    targets = render_gaussian_heatmap_targets(
        modes_list=modes_list,
        batch_size=1,
        feat_h=feat_h,
        feat_w=feat_w,
        stride=stride,
        device=device,
        render_sigma=2.0,
        u_observers=5,
        viewport_size_hw=(12, 20),
    )

    Y1 = targets["Y1"]
    Y_minus = targets["Y_minus"]
    mask_omega = targets["mask_omega"]
    has_aux = targets["has_aux"]

    assert has_aux[0].item() is True
    assert Y1.shape == (1, 1, feat_h, feat_w)
    assert Y_minus.shape == (1, 1, feat_h, feat_w)
    assert mask_omega.shape == (1, 1, feat_h, feat_w)

    # Top-1 center in feat coords: row 32 / 4 = 8, col 40 / 4 = 10
    # Peak at Top-1 should be 1.0 (support 5 / 5 = 1.0)
    assert torch.isclose(Y1.max(), torch.tensor(1.0), atol=1e-4)
    assert torch.isclose(Y1[0, 0, 8, 10], torch.tensor(1.0), atol=1e-4)
    # Peak at Top-2 should be 0.6 (support 3 / 5 = 0.6), at row 80/4=20, col 88/4=22
    assert torch.isclose(Y_minus.max(), torch.tensor(0.6), atol=1e-4)
    assert torch.isclose(Y_minus[0, 0, 20, 22], torch.tensor(0.6), atol=1e-4)
    # Top-1 mode must not leak into the auxiliary target
    assert Y_minus[0, 0, 8, 10].item() < 1e-6

    # Primary region mask A_t^(1): around (c1_y=32, c1_x=40) with h=12, w=20
    # -> y in [26, 38], x in [30, 50]
    # Inside primary region, mask_omega must be 0
    assert mask_omega[0, 0, 7, 10].item() == 0.0  # y=28, x=40
    assert mask_omega[0, 0, 8, 10].item() == 0.0  # y=32, x=40 (center)
    # Far away from primary region, mask_omega must be 1
    assert mask_omega[0, 0, 20, 20].item() == 1.0  # y=80, x=80


def test_l_rmc_empty_when_no_auxiliary_modes():
    # Only 1 mode (no auxiliary modes)
    modes_list = [
        {
            "centers": np.array([[30.0, 40.0]]),
            "support": np.array([5]),
            "n_observers": 5,
        }
    ]
    targets = render_gaussian_heatmap_targets(
        modes_list=modes_list,
        batch_size=1,
        feat_h=32,
        feat_w=32,
        stride=4,
        device=torch.device("cpu"),
    )

    pred_hm = torch.rand((1, 1, 32, 32))
    l_rmc = ranked_mode_coverage_loss(
        pred_hm=pred_hm,
        target_hm_minus=targets["Y_minus"],
        mask_omega=targets["mask_omega"],
        has_aux=targets["has_aux"],
    )
    assert l_rmc.item() == 0.0


def test_l_rmc_masks_out_primary_region():
    # 2 modes
    modes_list = [
        {
            "centers": np.array([[30.0, 40.0], [80.0, 90.0]]),
            "support": np.array([5, 3]),
            "n_observers": 5,
        }
    ]
    targets = render_gaussian_heatmap_targets(
        modes_list=modes_list,
        batch_size=1,
        feat_h=32,
        feat_w=32,
        stride=4,
        device=torch.device("cpu"),
    )

    # If pred_hm perfectly equals Y_minus everywhere outside primary region,
    # but has a huge error INSIDE primary region, L_rmc should still be near 0.0!
    pred_hm = targets["Y_minus"].clone()
    # Inject huge error inside primary region where mask_omega == 0
    pred_hm[targets["mask_omega"] == 0] += 999.0

    l_rmc = ranked_mode_coverage_loss(
        pred_hm=pred_hm,
        target_hm_minus=targets["Y_minus"],
        mask_omega=targets["mask_omega"],
        has_aux=targets["has_aux"],
    )
    assert torch.isclose(l_rmc, torch.tensor(0.0), atol=1e-5)


def test_l_hcm_focal_loss_backward():
    pred_hm = torch.tensor([[[[0.9, 0.1], [0.1, 0.1]]]], requires_grad=True)
    target_hm_top1 = torch.tensor([[[[1.0, 0.05], [0.05, 0.0]]]])
    pos_mask = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])

    loss = human_consensus_match_loss(pred_hm, target_hm_top1, pos_mask)
    assert loss.item() > 0.0
    loss.backward()
    assert pred_hm.grad is not None
    assert torch.isfinite(pred_hm.grad).all()


def test_l_rep_spatial_repulsion():
    # Box 1 and Box 2 non-overlapping
    b_no_overlap = torch.tensor([
        [0.0, 0.0, 10.0, 10.0],
        [20.0, 20.0, 30.0, 30.0],
    ], requires_grad=True)
    loss_no_overlap = spatial_repulsion_loss([b_no_overlap])
    assert loss_no_overlap.item() == 0.0

    # Box 1 and Box 2 completely overlapping (IoU = 1.0)
    b_overlap = torch.tensor([
        [10.0, 10.0, 20.0, 20.0],
        [10.0, 10.0, 20.0, 20.0],
    ], requires_grad=True)
    loss_overlap = spatial_repulsion_loss([b_overlap])
    assert torch.isclose(loss_overlap, torch.tensor(1.0), atol=1e-4)

    # Gradient check
    loss_overlap.backward()
    assert b_overlap.grad is not None


def test_l_smooth_trajectory():
    c_t = torch.tensor([[50.0, 50.0]], requires_grad=True)
    c_next_same = torch.tensor([[50.0, 50.0]])
    loss_zero = trajectory_smoothness_loss(c_t, c_next_same)
    assert loss_zero.item() == 0.0

    c_next_diff = torch.tensor([[53.0, 54.0]])  # dx=3, dy=4 -> dist_sq = 25
    loss_diff = trajectory_smoothness_loss(c_t, c_next_diff)
    assert torch.isclose(loss_diff, torch.tensor(25.0), atol=1e-4)

    loss_diff.backward()
    assert c_t.grad is not None


if __name__ == "__main__":
    test_render_targets_and_masking()
    print("test_render_targets_and_masking: PASS")
    test_l_rmc_empty_when_no_auxiliary_modes()
    print("test_l_rmc_empty_when_no_auxiliary_modes: PASS")
    test_l_rmc_masks_out_primary_region()
    print("test_l_rmc_masks_out_primary_region: PASS")
    test_l_hcm_focal_loss_backward()
    print("test_l_hcm_focal_loss_backward: PASS")
    test_l_rep_spatial_repulsion()
    print("test_l_rep_spatial_repulsion: PASS")
    test_l_smooth_trajectory()
    print("test_l_smooth_trajectory: PASS")
    print("\nALL DIRECTOR LOSS UNIT TESTS PASSED!")
