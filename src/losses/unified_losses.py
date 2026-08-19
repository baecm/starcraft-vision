# src/losses/unified_losses.py
from __future__ import annotations

from typing import Dict, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_spatial_entropy_map(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Computes Self-Supervised Spatial Entropy Map S_entropy(x,y,t) from state tensor X_t.
    x: State tensor of shape (B, T, C, H, W)
    Returns: Spatial entropy map of shape (B, T, 1, H, W) normalized to [0, 1]
    """
    B, T, C, H, W = x.shape
    # Softmax across channels C to form probability distribution over features per pixel
    p = F.softmax(x, dim=2)  # (B, T, C, H, W)
    entropy = -torch.sum(p * torch.log(p + eps), dim=2, keepdim=True)  # (B, T, 1, H, W)

    # Normalize map to [0, 1] per frame
    ent_flat = entropy.view(B * T, 1, H * W)
    min_val = ent_flat.min(dim=-1, keepdim=True)[0]
    max_val = ent_flat.max(dim=-1, keepdim=True)[0]
    norm_entropy = (ent_flat - min_val) / (max_val - min_val + eps)

    return norm_entropy.view(B, T, 1, H, W)


def compute_cvae_kl_loss(
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mu_p: torch.Tensor,
    logvar_p: torch.Tensor,
) -> torch.Tensor:
    """
    Computes KL Divergence D_KL( q(z|F, H) || p(z|F) ) between Gaussian distributions.
    Shapes: (B, T, D) or (B, D).
    """
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)

    kl = 0.5 * (
        logvar_p - logvar_q + (var_q + (mu_q - mu_p) ** 2) / (var_p + 1e-8) - 1.0
    )
    return kl.sum(dim=-1).mean()


class HysteresisAntiThrashingLoss(nn.Module):
    """
    Hysteresis Anti-Thrashing Loss Cost_thrashing(Y_t, Y_{t-1}).
    Penalizes viewport velocity and acceleration outside a deadzone threshold delta_deadzone.
    """

    def __init__(self, deadzone: float = 0.03, weight_jerk: float = 0.5):
        super().__init__()
        self.deadzone = deadzone
        self.weight_jerk = weight_jerk

    def forward(self, pred_boxes: torch.Tensor) -> torch.Tensor:
        """
        pred_boxes: (B, T, K, 4) in normalized [cx, cy, w, h] format.
        """
        B, T, K, _ = pred_boxes.shape
        if T < 2:
            return torch.tensor(0.0, device=pred_boxes.device)

        # Extract viewport centers (B, T, K, 2)
        centers = pred_boxes[..., :2]

        # Velocity: p_t - p_{t-1}
        vel = centers[:, 1:] - centers[:, :-1]  # (B, T-1, K, 2)
        disp = torch.norm(vel, dim=-1)           # (B, T-1, K)

        # Apply deadzone hysteresis penalty
        penalty_disp = torch.clamp(disp - self.deadzone, min=0.0) ** 2
        loss_vel = penalty_disp.mean()

        if T >= 3:
            # Acceleration: v_t - v_{t-1}
            acc = vel[:, 1:] - vel[:, :-1]  # (B, T-2, K, 2)
            if T >= 4:
                # Jerk: a_t - a_{t-1}
                jerk = acc[:, 1:] - acc[:, :-1]  # (B, T-3, K, 2)
                loss_jerk = (torch.norm(jerk, dim=-1) ** 2).mean()
            else:
                loss_jerk = (torch.norm(acc, dim=-1) ** 2).mean()
        else:
            loss_jerk = torch.tensor(0.0, device=pred_boxes.device)

        return loss_vel + self.weight_jerk * loss_jerk


class ConsensusWeightedLoss(nn.Module):
    """
    Track A Consensus-Weighted Loss:
    Combines Heatmap Regression Loss against 5-rater consensus density map W_consensus,
    CVAE KL Divergence loss, and smooth L1 box regression.
    """

    def __init__(
        self,
        weight_heatmap: float = 1.0,
        weight_kl: float = 0.1,
        weight_box: float = 1.0,
    ):
        super().__init__()
        self.weight_heatmap = weight_heatmap
        self.weight_kl = weight_kl
        self.weight_box = weight_box

    def forward(
        self,
        pred_heatmap: torch.Tensor,
        consensus_map: torch.Tensor,
        latent_out: Dict[str, torch.Tensor],
        pred_boxes: Optional[torch.Tensor] = None,
        target_boxes: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        pred_heatmap: (B, T, 1, H, W)
        consensus_map: (B, T, 1, H, W) or (B, T, H, W)
        latent_out: Dict containing mu, logvar, prior_mu, prior_logvar
        """
        if consensus_map.ndim == 4:
            consensus_map = consensus_map.unsqueeze(2)

        # Heatmap regression loss weighted by consensus map intensity
        h_loss = F.mse_loss(pred_heatmap, consensus_map)

        # CVAE KL divergence loss
        kl_loss = compute_cvae_kl_loss(
            mu_q=latent_out["mu"],
            logvar_q=latent_out["logvar"],
            mu_p=latent_out["prior_mu"],
            logvar_p=latent_out["prior_logvar"],
        )

        box_loss = torch.tensor(0.0, device=pred_heatmap.device)
        if pred_boxes is not None and target_boxes is not None:
            box_loss = F.smooth_l1_loss(pred_boxes, target_boxes)

        total_track_a = (
            self.weight_heatmap * h_loss
            + self.weight_kl * kl_loss
            + self.weight_box * box_loss
        )

        return {
            "loss_track_a": total_track_a,
            "loss_heatmap": h_loss,
            "loss_kl": kl_loss,
            "loss_box": box_loss,
        }


class TrackBUnsupervisedLoss(nn.Module):
    """
    Track B Zero-Shot Loss (Unsupervised Constraints):
      1. Information Coverage Loss (maximizes spatial entropy map coverage)
      2. Pairwise Redundancy Suppression Loss (penalizes viewport overlap)
      3. Spatio-Temporal Continuity Loss (anti-thrashing)
    """

    def __init__(
        self,
        weight_coverage: float = 1.0,
        weight_redundancy: float = 1.0,
        weight_continuity: float = 0.5,
    ):
        super().__init__()
        self.weight_coverage = weight_coverage
        self.weight_redundancy = weight_redundancy
        self.weight_continuity = weight_continuity
        self.anti_thrashing = HysteresisAntiThrashingLoss()

    def forward(
        self,
        pred_boxes: torch.Tensor,
        spatial_entropy_map: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        pred_boxes: (B, T, K, 4) in [cx, cy, w, h] normalized format.
        spatial_entropy_map: (B, T, 1, H, W) normalized entropy.
        """
        B, T, K, _ = pred_boxes.shape
        H, W = spatial_entropy_map.shape[-2:]

        # 1. Information Coverage Loss: Encourage viewports to cover high entropy regions
        # Generate grid coordinates [0, 1]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, H, device=pred_boxes.device),
            torch.linspace(0, 1, W, device=pred_boxes.device),
            indexing="ij",
        )  # (H, W)

        coverage_loss_list = []
        for b in range(B):
            for t in range(T):
                ent_map = spatial_entropy_map[b, t, 0]  # (H, W)
                boxes_bt = pred_boxes[b, t]              # (K, 4)

                # Soft mask for K viewports
                union_soft_mask = torch.zeros_like(ent_map)
                for k in range(K):
                    cx, cy, w, h = boxes_bt[k]
                    x1, y1 = cx - w / 2.0, cy - h / 2.0
                    x2, y2 = cx + w / 2.0, cy + h / 2.0

                    # Soft box mask via sigmoid temperature
                    mask_x = torch.sigmoid(10.0 * (grid_x - x1)) * torch.sigmoid(10.0 * (x2 - grid_x))
                    mask_y = torch.sigmoid(10.0 * (grid_y - y1)) * torch.sigmoid(10.0 * (y2 - grid_y))
                    soft_box = mask_x * mask_y
                    union_soft_mask = torch.max(union_soft_mask, soft_box)

                covered_entropy = torch.sum(union_soft_mask * ent_map)
                total_entropy = torch.sum(ent_map) + 1e-8
                coverage_loss_list.append(-torch.log(covered_entropy / total_entropy + 1e-8))

        loss_coverage = torch.stack(coverage_loss_list).mean()

        # 2. Pairwise Redundancy Suppression Loss (penalize center closeness)
        centers = pred_boxes[..., :2]  # (B, T, K, 2)
        pairwise_dist = torch.norm(centers.unsqueeze(3) - centers.unsqueeze(2), dim=-1)  # (B, T, K, K)
        # Mask out self distance
        eye = torch.eye(K, device=pred_boxes.device).view(1, 1, K, K)
        pairwise_dist = pairwise_dist + eye * 1e5
        # Penalize distance smaller than threshold d_min
        d_min = 0.2
        loss_redundancy = torch.clamp(d_min - pairwise_dist, min=0.0).mean()

        # 3. Spatio-Temporal Continuity Loss
        loss_continuity = self.anti_thrashing(pred_boxes)

        total_track_b = (
            self.weight_coverage * loss_coverage
            + self.weight_redundancy * loss_redundancy
            + self.weight_continuity * loss_continuity
        )

        return {
            "loss_track_b": total_track_b,
            "loss_coverage": loss_coverage,
            "loss_redundancy": loss_redundancy,
            "loss_continuity": loss_continuity,
        }


class UnifiedEnergyLoss(nn.Module):
    """
    Unified Energy Objective Loss:
        Loss = alpha * Loss_state (Track B) + beta * Loss_human (Track A) + lambda * Cost_thrashing

    Enables joint training or flexible switching between Track A (Human consensus supervision)
    and Track B (Zero-shot self-supervised state entropy supervision).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        lambd: float = 0.5,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.lambd = lambd

        self.track_a_loss = ConsensusWeightedLoss()
        self.track_b_loss = TrackBUnsupervisedLoss()
        self.anti_thrashing = HysteresisAntiThrashingLoss()

    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        state_tensor: torch.Tensor,
        consensus_map: Optional[torch.Tensor] = None,
        target_boxes: Optional[torch.Tensor] = None,
        mode: str = "combined",
    ) -> Dict[str, torch.Tensor]:
        """
        model_outputs: Output dict from ProbabilisticVideoDETR
        state_tensor: Input state tensor (B, T, C, H, W)
        consensus_map: Optional 5-rater consensus density map (B, T, 1, H, W)
        mode: 'track_a', 'track_b', or 'combined'
        """
        pred_boxes = model_outputs["pred_boxes"]
        pred_heatmap = model_outputs["pred_heatmap"]
        latent_out = model_outputs["latent_out"]

        total_loss = torch.tensor(0.0, device=pred_boxes.device)
        loss_dict: Dict[str, torch.Tensor] = {}

        thrashing_cost = self.anti_thrashing(pred_boxes)
        loss_dict["cost_thrashing"] = thrashing_cost

        if mode in ("track_a", "combined") and consensus_map is not None:
            a_dict = self.track_a_loss(
                pred_heatmap=pred_heatmap,
                consensus_map=consensus_map,
                latent_out=latent_out,
                pred_boxes=pred_boxes,
                target_boxes=target_boxes,
            )
            loss_dict.update(a_dict)
            total_loss = total_loss + self.beta * a_dict["loss_track_a"]

        if mode in ("track_b", "combined"):
            spatial_entropy = compute_spatial_entropy_map(state_tensor)
            b_dict = self.track_b_loss(
                pred_boxes=pred_boxes, spatial_entropy_map=spatial_entropy
            )
            loss_dict.update(b_dict)
            total_loss = total_loss + self.alpha * b_dict["loss_track_b"]

        total_loss = total_loss + self.lambd * thrashing_cost
        loss_dict["total_loss"] = total_loss

        return loss_dict
