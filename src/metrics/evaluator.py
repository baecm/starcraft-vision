# src/metrics/evaluator.py
from __future__ import annotations

from typing import Dict, List, Tuple, Union, Optional, Any
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def _to_numpy(data: Union[torch.Tensor, np.ndarray, List[Any]]) -> np.ndarray:
    """Helper to convert input torch.Tensor or list to np.ndarray."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    if isinstance(data, list):
        return np.array(data, dtype=np.float32)
    return np.asarray(data, dtype=np.float32)


def box_cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Converts boxes from [cx, cy, w, h] format to [x1, y1, x2, y2]."""
    if boxes.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return np.stack([x1, y1, x2, y2], axis=-1)


def compute_iou_boxes(boxA: np.ndarray, boxB: np.ndarray) -> float:
    """Computes Intersection over Union (IoU) between two 1D box arrays [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_area = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union_area = boxA_area + boxB_area - inter_area

    if union_area <= 1e-8:
        return 0.0
    return float(inter_area / union_area)


def compute_cwo(
    viewports: Union[torch.Tensor, np.ndarray],
    consensus_map: Union[torch.Tensor, np.ndarray],
    grid_size: Tuple[int, int] = (128, 128),
    box_format: str = "xyxy",
) -> float:
    """
    Computes Consensus-Weighted Overlap (CWO):
        CWO(Y_t) = \int_{\bigcup v_i} W_consensus(x, y) dx dy / \int_{\Omega} W_consensus(x, y) dx dy

    Args:
        viewports: Array/Tensor of shape (K, 4) representing K viewports.
        consensus_map: 2D array/tensor of shape (H, W) representing consensus density.
        grid_size: Target (H, W) grid dimensions for rasterization if maps are normalized.
        box_format: 'xyxy' or 'cxcywh'.

    Returns:
        float CWO score in range [0.0, 1.0].
    """
    v_arr = _to_numpy(viewports)
    c_map = _to_numpy(consensus_map)

    if c_map.ndim != 2:
        raise ValueError(f"consensus_map must be 2D array, got shape {c_map.shape}")

    H, W = c_map.shape
    total_consensus = float(c_map.sum())
    if total_consensus <= 1e-8:
        return 0.0

    if v_arr.size == 0 or len(v_arr) == 0:
        return 0.0

    if box_format == "cxcywh":
        v_arr = box_cxcywh_to_xyxy(v_arr)

    # Rasterize union mask of K viewports
    union_mask = np.zeros((H, W), dtype=bool)
    grid_y, grid_x = np.ogrid[:H, :W]

    for box in v_arr:
        x1, y1, x2, y2 = box
        # Scale if normalized in range [0, 1]
        if max(x2, y2) <= 1.0 and min(x1, y1) >= 0.0:
            x1, x2 = x1 * W, x2 * W
            y1, y2 = y1 * H, y2 * H

        x1_i, x2_i = int(np.clip(np.floor(x1), 0, W)), int(np.clip(np.ceil(x2), 0, W))
        y1_i, y2_i = int(np.clip(np.floor(y1), 0, H)), int(np.clip(np.ceil(y2), 0, H))

        if x2_i > x1_i and y2_i > y1_i:
            union_mask[y1_i:y2_i, x1_i:x2_i] = True

    covered_consensus = float(c_map[union_mask].sum())
    return covered_consensus / total_consensus


def compute_m_cti(
    trajectories: Union[torch.Tensor, np.ndarray, List[np.ndarray]],
    jump_threshold: float = 35.0,
    w_jerk: float = 0.4,
    w_jump: float = 0.4,
    w_vel: float = 0.2,
    box_format: str = "xyxy",
) -> Dict[str, float]:
    """
    Computes Multi-Track Camera Thrashing Index (M-CTI) over a temporal sequence.
    Uses Hungarian Matching to track identity across frame transitions and evaluates:
      - Mean Velocity
      - Mean Jerk (derivative of acceleration)
      - Jump Rate (teleport jump ratio where displacement > jump_threshold)

    Args:
        trajectories: Sequence of viewports across T frames. Shape (T, K, 4) or list of T arrays (K_t, 4).
        jump_threshold: Distance threshold for frame-to-frame jump/teleport.
        w_jerk, w_jump, w_vel: Metric weighting parameters.
        box_format: 'xyxy' or 'cxcywh'.

    Returns:
        Dict with keys: 'm_cti', 'jerk', 'jump_rate', 'velocity'.
    """
    if isinstance(trajectories, list):
        seq = [_to_numpy(t) for t in trajectories]
    else:
        seq = [_to_numpy(trajectories[t]) for t in range(len(trajectories))]

    T = len(seq)
    if T < 2:
        return {"m_cti": 0.0, "jerk": 0.0, "jump_rate": 0.0, "velocity": 0.0}

    # Extract centers for each frame
    centers_per_frame: List[np.ndarray] = []
    for frame_boxes in seq:
        if frame_boxes.size == 0:
            centers_per_frame.append(np.empty((0, 2), dtype=np.float32))
            continue
        if box_format == "cxcywh":
            pts = frame_boxes[:, :2]
        else:
            cx = (frame_boxes[:, 0] + frame_boxes[:, 2]) / 2.0
            cy = (frame_boxes[:, 1] + frame_boxes[:, 3]) / 2.0
            pts = np.stack([cx, cy], axis=-1)
        centers_per_frame.append(pts)

    K = len(centers_per_frame[0])
    if K == 0:
        return {"m_cti": 0.0, "jerk": 0.0, "jump_rate": 0.0, "velocity": 0.0}

    # Hungarian matching trajectory identity tracking
    tracked_centers = np.zeros((T, K, 2), dtype=np.float32)
    tracked_centers[0] = centers_per_frame[0]

    for t in range(1, T):
        prev_pts = tracked_centers[t - 1]
        curr_pts = centers_per_frame[t]

        if len(curr_pts) != K:
            # Fallback if variable K per frame
            num_match = min(K, len(curr_pts))
            tracked_centers[t, :num_match] = curr_pts[:num_match]
            continue

        # Cost matrix: pairwise Euclidean distance
        cost_matrix = np.linalg.norm(prev_pts[:, None, :] - curr_pts[None, :, :], axis=-1)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        tracked_centers[t, row_ind] = curr_pts[col_ind]

    # Compute Velocity: v_t = p_t - p_{t-1}
    vel = np.diff(tracked_centers, axis=0)  # Shape (T-1, K, 2)
    speed = np.linalg.norm(vel, axis=-1)   # Shape (T-1, K)

    mean_velocity = float(speed.mean())

    # Compute Jumps
    jumps = speed > jump_threshold
    jump_rate = float(jumps.sum() / max(1, speed.size))

    # Compute Acceleration: a_t = v_t - v_{t-1}
    if T >= 3:
        acc = np.diff(vel, axis=0)  # Shape (T-2, K, 2)
        # Compute Jerk: j_t = a_t - a_{t-1}
        if T >= 4:
            jerk = np.diff(acc, axis=0)  # Shape (T-3, K, 2)
            jerk_norm = np.linalg.norm(jerk, axis=-1)
            mean_jerk = float(jerk_norm.mean())
        else:
            acc_norm = np.linalg.norm(acc, axis=-1)
            mean_jerk = float(acc_norm.mean())
    else:
        mean_jerk = 0.0

    m_cti_score = (w_jerk * mean_jerk) + (w_jump * jump_rate) + (w_vel * mean_velocity)

    return {
        "m_cti": float(m_cti_score),
        "jerk": float(mean_jerk),
        "jump_rate": float(jump_rate),
        "velocity": float(mean_velocity),
    }


def compute_event_recall(
    viewports: Union[torch.Tensor, np.ndarray],
    event_coords: Union[torch.Tensor, np.ndarray, List[Tuple[float, float]]],
    box_format: str = "xyxy",
) -> float:
    """
    Computes Objective Event Recall (R_event):
        Percentage of objective battle/event coordinates captured within at least one viewport.

    Args:
        viewports: Array/Tensor of shape (K, 4) for viewports.
        event_coords: Array/Tensor of shape (N, 2) representing event (x, y) points.
        box_format: 'xyxy' or 'cxcywh'.

    Returns:
        float event recall score in range [0.0, 1.0].
    """
    v_arr = _to_numpy(viewports)
    e_arr = _to_numpy(event_coords)

    if e_arr.size == 0 or len(e_arr) == 0:
        return 1.0  # No events present

    if v_arr.size == 0 or len(v_arr) == 0:
        return 0.0

    if box_format == "cxcywh":
        v_arr = box_cxcywh_to_xyxy(v_arr)

    # Check for each event if it lands in any viewport
    recalled_count = 0
    for ex, ey in e_arr:
        captured = False
        for box in v_arr:
            x1, y1, x2, y2 = box
            if x1 <= ex <= x2 and y1 <= ey <= y2:
                captured = True
                break
        if captured:
            recalled_count += 1

    return float(recalled_count / len(e_arr))


def compute_pairwise_overlap(
    viewports: Union[torch.Tensor, np.ndarray],
    box_format: str = "xyxy",
) -> float:
    """
    Computes mean pairwise IoU among K viewports to evaluate spatial redundancy.

    Args:
        viewports: Array/Tensor of shape (K, 4).
        box_format: 'xyxy' or 'cxcywh'.

    Returns:
        float pairwise overlap score in range [0.0, 1.0].
    """
    v_arr = _to_numpy(viewports)
    K = len(v_arr)
    if K < 2:
        return 0.0

    if box_format == "cxcywh":
        v_arr = box_cxcywh_to_xyxy(v_arr)

    total_iou = 0.0
    num_pairs = 0
    for i in range(K):
        for j in range(i + 1, K):
            total_iou += compute_iou_boxes(v_arr[i], v_arr[j])
            num_pairs += 1

    return float(total_iou / max(1, num_pairs))


class MultiRegionEvaluator:
    """
    Unified 3D Evaluation Suite Evaluator for Multi-Region Finding & Multi-Viewport Control:
      - Consensus-Weighted Overlap (CWO)
      - Multi-Track Camera Thrashing Index (M-CTI)
      - Objective Event Recall (R_event)
      - Pairwise Viewport Overlap
    """

    def __init__(
        self,
        grid_size: Tuple[int, int] = (128, 128),
        jump_threshold: float = 35.0,
        box_format: str = "xyxy",
        m_cti_weights: Tuple[float, float, float] = (0.4, 0.4, 0.2),
    ):
        self.grid_size = grid_size
        self.jump_threshold = jump_threshold
        self.box_format = box_format
        self.w_jerk, self.w_jump, self.w_vel = m_cti_weights

    def evaluate_frame(
        self,
        viewports: Union[torch.Tensor, np.ndarray],
        consensus_map: Optional[Union[torch.Tensor, np.ndarray]] = None,
        event_coords: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> Dict[str, float]:
        """Evaluates single frame viewports against consensus map and objective events."""
        results: Dict[str, float] = {}

        results["pairwise_overlap"] = compute_pairwise_overlap(
            viewports, box_format=self.box_format
        )

        if consensus_map is not None:
            results["cwo"] = compute_cwo(
                viewports, consensus_map, grid_size=self.grid_size, box_format=self.box_format
            )

        if event_coords is not None:
            results["event_recall"] = compute_event_recall(
                viewports, event_coords, box_format=self.box_format
            )

        return results

    def evaluate_sequence(
        self,
        viewports_seq: Union[List[np.ndarray], torch.Tensor],
        consensus_maps_seq: Optional[Union[List[np.ndarray], torch.Tensor]] = None,
        event_coords_seq: Optional[List[Union[np.ndarray, torch.Tensor]]] = None,
    ) -> Dict[str, float]:
        """
        Evaluates sequence of frames over time and aggregates 3D Evaluation metrics.

        Args:
            viewports_seq: List of length T with shape (K, 4) viewports per frame.
            consensus_maps_seq: Optional list of length T with 2D consensus maps.
            event_coords_seq: Optional list of length T with (N_t, 2) event coordinates per frame.

        Returns:
            Dictionary containing averaged CWO, R_event, Pairwise Overlap, and sequence M-CTI.
        """
        T = len(viewports_seq)
        cwo_list = []
        recall_list = []
        overlap_list = []

        for t in range(T):
            v_t = viewports_seq[t]
            c_t = consensus_maps_seq[t] if consensus_maps_seq is not None else None
            e_t = event_coords_seq[t] if event_coords_seq is not None else None

            frame_metrics = self.evaluate_frame(
                viewports=v_t, consensus_map=c_t, event_coords=e_t
            )

            if "cwo" in frame_metrics:
                cwo_list.append(frame_metrics["cwo"])
            if "event_recall" in frame_metrics:
                recall_list.append(frame_metrics["event_recall"])
            overlap_list.append(frame_metrics["pairwise_overlap"])

        # Sequence-level M-CTI
        m_cti_metrics = compute_m_cti(
            trajectories=viewports_seq,
            jump_threshold=self.jump_threshold,
            w_jerk=self.w_jerk,
            w_jump=self.w_jump,
            w_vel=self.w_vel,
            box_format=self.box_format,
        )

        aggregated: Dict[str, float] = {
            "cwo": float(np.mean(cwo_list)) if cwo_list else 0.0,
            "event_recall": float(np.mean(recall_list)) if recall_list else 0.0,
            "pairwise_overlap": float(np.mean(overlap_list)) if overlap_list else 0.0,
            "m_cti": m_cti_metrics["m_cti"],
            "jerk": m_cti_metrics["jerk"],
            "jump_rate": m_cti_metrics["jump_rate"],
            "velocity": m_cti_metrics["velocity"],
        }

        return aggregated
