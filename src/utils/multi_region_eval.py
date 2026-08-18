# src/utils/multi_region_eval.py
from __future__ import annotations

from typing import Dict, List, Tuple, Any
import numpy as np


def compute_multi_region_metrics(
    coco_gt: Any,
    preds_list: List[dict],
    grid_w: int = 128,
    grid_h: int = 128,
    win_w: int = 20,
    win_h: int = 12,
    top_ks: Tuple[int, ...] = (1, 3, 5),
    soft_radii: Tuple[float, ...] = (10.0, 20.0),
) -> Dict[str, float]:
    """
    Computes a comprehensive set of Multi-Region Finding & GT-Noise-Robust metrics:
      1. Top-K Cumulative Intersection Coverage (top3_ic@050, top5_ic@050)
      2. Cluster Recall (GT Spatial Clustering -> % clusters covered by Top-K predictions)
      3. Min GT Distance (Nearest GT cluster distance in grid tiles)
      4. Soft-Radius Precision & Recall (Prec@R10, Recall@R10, Prec@R20, Recall@R20)
      5. Inter-Region Spatial Dispersion (Average pairwise distance between Top-K regions)
    """
    preds_by_img: Dict[int, List[dict]] = {}
    for p in preds_list:
        img_id = int(p["image_id"])
        preds_by_img.setdefault(img_id, []).append(p)

    topk_coverages = {k: [] for k in top_ks}
    cluster_recalls = {k: [] for k in top_ks}
    min_gt_dists = {k: [] for k in top_ks}
    prec_r = {r: [] for r in soft_radii}
    recall_r = {r: [] for r in soft_radii}
    inter_region_dists = []

    for img in coco_gt.dataset.get("images", []):
        image_id = int(img["id"])
        img_w = int(img.get("width", grid_w))
        img_h = int(img.get("height", grid_h))

        ann_ids = coco_gt.getAnnIds(imgIds=image_id)
        anns = coco_gt.loadAnns(ann_ids) if ann_ids else []
        if not anns:
            continue

        gt_points = []
        for ann in anns:
            if "bbox" in ann and ann["bbox"]:
                x, y, w, h = ann["bbox"]
                cx, cy = x + w / 2.0, y + h / 2.0
            else:
                cx, cy = img_w / 2.0, img_h / 2.0
            gx = (cx / max(1, img_w)) * grid_w
            gy = (cy / max(1, img_h)) * grid_h
            gt_points.append([gx, gy])

        gt_pts_arr = np.array(gt_points, dtype=np.float32)

        clusters = []
        visited = set()
        cluster_radius = 15.0
        for i in range(len(gt_points)):
            if i in visited:
                continue
            dists = np.linalg.norm(gt_pts_arr - gt_pts_arr[i], axis=1)
            members = np.where(dists <= cluster_radius)[0]
            visited.update(members)
            clusters.append(gt_pts_arr[members].mean(axis=0))

        clusters_arr = np.array(clusters, dtype=np.float32) if clusters else gt_pts_arr

        img_preds = preds_by_img.get(image_id, [])
        img_preds = sorted(img_preds, key=lambda q: float(q.get("score", 0.0)), reverse=True)

        if not img_preds:
            continue

        pred_centers = []
        for p in img_preds[:max(top_ks)]:
            if "bbox" in p and p["bbox"]:
                x, y, w, h = p["bbox"]
                cx, cy = x + w / 2.0, y + h / 2.0
            else:
                cx, cy = img_w / 2.0, img_h / 2.0
            px = (cx / max(1, img_w)) * grid_w
            py = (cy / max(1, img_h)) * grid_h
            pred_centers.append([px, py])

        pred_pts_arr = np.array(pred_centers, dtype=np.float32)

        half_w, half_h = win_w / 2.0, win_h / 2.0

        for k in top_ks:
            curr_preds = pred_pts_arr[:k]
            if len(curr_preds) == 0:
                topk_coverages[k].append(0.0)
                cluster_recalls[k].append(0.0)
                min_gt_dists[k].append(float(grid_w))
                continue

            covered_units = 0
            for gt_pt in gt_pts_arr:
                dx = np.abs(curr_preds[:, 0] - gt_pt[0])
                dy = np.abs(curr_preds[:, 1] - gt_pt[1])
                if np.any((dx <= half_w) & (dy <= half_h)):
                    covered_units += 1
            topk_coverages[k].append(covered_units / max(1, len(gt_pts_arr)))

            covered_clusters = 0
            for clus in clusters_arr:
                dx = np.abs(curr_preds[:, 0] - clus[0])
                dy = np.abs(curr_preds[:, 1] - clus[1])
                if np.any((dx <= half_w) & (dy <= half_h)):
                    covered_clusters += 1
            cluster_recalls[k].append(covered_clusters / max(1, len(clusters_arr)))

            dists_matrix = np.linalg.norm(curr_preds[:, None, :] - clusters_arr[None, :, :], axis=2)
            min_dists = dists_matrix.min(axis=1)
            min_gt_dists[k].append(float(min_dists.mean()))

        top3_preds = pred_pts_arr[:3]
        for r in soft_radii:
            if len(top3_preds) == 0:
                prec_r[r].append(0.0)
                recall_r[r].append(0.0)
                continue
            dists_matrix = np.linalg.norm(top3_preds[:, None, :] - gt_pts_arr[None, :, :], axis=2)
            prec = np.mean(np.any(dists_matrix <= r, axis=1))
            rec = np.mean(np.any(dists_matrix <= r, axis=0))
            prec_r[r].append(float(prec))
            recall_r[r].append(float(rec))

        if len(top3_preds) > 1:
            diff = top3_preds[:, None, :] - top3_preds[None, :, :]
            pair_dists = np.linalg.norm(diff, axis=2)
            triu_idx = np.triu_indices(len(top3_preds), k=1)
            inter_region_dists.append(float(pair_dists[triu_idx].mean()))

    res = {}
    for k in top_ks:
        res[f"top{k}_ic@050"] = float(np.mean(topk_coverages[k])) if topk_coverages[k] else 0.0
        res[f"cluster_recall@top{k}"] = float(np.mean(cluster_recalls[k])) if cluster_recalls[k] else 0.0
        res[f"min_gt_dist_top{k}"] = float(np.mean(min_gt_dists[k])) if min_gt_dists[k] else 0.0

    for r in soft_radii:
        r_int = int(r)
        res[f"prec_r{r_int}"] = float(np.mean(prec_r[r])) if prec_r[r] else 0.0
        res[f"recall_r{r_int}"] = float(np.mean(recall_r[r])) if recall_r[r] else 0.0

    res["inter_region_dist"] = float(np.mean(inter_region_dists)) if inter_region_dists else 0.0
    return res
