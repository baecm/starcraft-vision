# /src/evaluate.py

import os
import json
import argparse
import re
import hashlib
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from collections import OrderedDict

import numpy as np
import pandas as pd
import config
from utils.logger import Logger  # Logger 사용


# =========================
# Fixed sets
# =========================
SET_REPLAYS = {
    "set_0": ["36", "212", "438", "522", "1660"],
    "set_1": ["1559", "1628", "2351", "6219", "11251"],
    "set_2": ["275", "1725", "3613", "4520", "4664"]
}

SET_GT_ANNOTATORS = {
    "set_0": ["bcm_allframes", "yws_allframes", "cyh_allframes", "pdh_allframes", "jht_allframes"],
    "set_1": ["1_allframes", "2_allframes", "3_allframes", "4_allframes", "5_allframes"],
    "set_2": ["6_allframes", "7_allframes", "8_allframes", "9_allframes", "10_allframes"]
}


# =========================
# Grid config (SSOT)
# =========================
@dataclass(frozen=True)
class Grid:
    x_len: int = 20
    y_len: int = 12
    width: int = 128
    height: int = 128
    max_x: int = 3456
    max_y: int = 3720

    @classmethod
    def from_args(cls, args) -> "Grid":
        return cls(
            x_len=args.kernel_x,
            y_len=args.kernel_y,
            width=args.grid_width,
            height=args.grid_height,
            max_x=args.max_x,
            max_y=args.max_y,
        )


def _assert_dir(path: Path, what: str):
    if not path.is_dir():
        Logger.error(f"{what} not found or not a directory: {path}")
        raise FileNotFoundError(f"{what} not found: {path}")


# =========================
# Helpers (time/slug/hash/keys)
# =========================
def _write_vpd(df: pd.DataFrame, out_path: Path, scale: int = 32,
               clip_max: Optional[Tuple[int, int]] = None):
    """
    df: columns include ['frame','x','y']
    Writes CSV with header: frame,vpx,vpy (ints), where vpx,vpy = round(x*scale), round(y*scale)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vpd = df[['frame', 'x', 'y']].copy()
    vpd['frame'] = vpd['frame'].astype(int)

    vpd['vpx'] = np.round(vpd['x'] * scale).astype(int)
    vpd['vpy'] = np.round(vpd['y'] * scale).astype(int)

    if clip_max is not None:
        max_x, max_y = clip_max
        vpd['vpx'] = vpd['vpx'].clip(lower=0, upper=max_x - 1)
        vpd['vpy'] = vpd['vpy'].clip(lower=0, upper=max_y - 1)

    vpd = vpd[['frame', 'vpx', 'vpy']]
    vpd.to_csv(out_path, index=False)
    Logger.info(f"Wrote VPD -> {out_path}")


def _dump_vpd_for_tracks(
    frame_select: str,
    replays: List[str],
    tracks: List[pd.DataFrame],
    vpd_root: Path,
    scale: int = 32,
    clip_max: Optional[Tuple[int, int]] = None
):
    """
    저장 경로: <vpd_root>/<frame_select>/<replay>.vpd
    """
    base = vpd_root / frame_select
    for rep, df in zip(replays, tracks):
        out_path = base / f"{rep}.vpd"
        _write_vpd(df, out_path, scale=scale, clip_max=clip_max)
        

def _now_utc_str() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _slugify(s: str) -> str:
    s = re.sub(r"\s+", "-", s.strip())
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", s).strip("-._")


def _short_hash(text: str, n: int = 8) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def _abbr_label(label: str) -> str:
    # e.g., "all_correct" -> "ac"
    parts = [p[0] for p in label.split("_") if p]
    return "".join(parts) or _slugify(label)[:6]


def _abbr_fs(fs: str) -> str:
    return {"first": "f", "top1": "t1", "avg": "a"}.get(fs, _slugify(fs)[:3])


def _fs_code(fs_list: List[str]) -> str:
    s = set(fs_list)
    if s == {"first", "top1", "avg"} and len(fs_list) == 3:
        return "all"
    return "-".join(_abbr_fs(x) for x in fs_list)[:12]


def _pct_code(p: float) -> str:
    return f"{int(round(p * 100)):d}"


def _replays_key(args) -> str:
    if getattr(args, "set", None):
        m = re.match(r"set_(\d+)", args.set)
        return f"s{m.group(1)}" if m else _slugify(args.set)[:8]
    return f"r{len(args.replays)}"


def _pred_key(pred_names: List[str]) -> str:
    if not pred_names:
        return "gtloo"
    if len(pred_names) == 1:
        return _slugify(Path(pred_names[0]).name)[:24]
    return f"multi{len(pred_names)}"


def _thr_key(t: float) -> str:
    # 0 -> "0", 0.1 -> "0_1", 0.75 -> "0_75"
    s = f"{t:.6f}".rstrip("0").rstrip(".")
    return s.replace(".", "_") if s else "0"

def _ordered_metric_keys(ic_thresholds, has_baseline=False, has_ratio=False):
    keys = []
    for t in sorted(set(ic_thresholds)):
        tk = _thr_key(t)  # 0.3 -> "0_3"
        keys += [f"ic_at_{tk}", f"streak_at_{tk}"]
        if has_baseline:
            keys.append(f"streak_baseline_at_{tk}")
        if has_ratio:
            keys.append(f"streak_ratio_at_{tk}")
    keys += ["ic_mean", "iou_mean", "dice_mean", "cr_mean"]
    return keys

def _order_row(row: dict, ic_thresholds) -> OrderedDict:
    has_baseline = any(k.startswith("streak_baseline_at_") for k in row)
    has_ratio    = any(k.startswith("streak_ratio_at_") for k in row)
    metric_keys  = _ordered_metric_keys(ic_thresholds, has_baseline, has_ratio)

    out = OrderedDict()
    # 식별자 먼저
    for k in ("target", "frame_select"):
        if k in row:
            out[k] = row[k]
    # 메트릭을 원하는 순서로
    for k in metric_keys:
        if k in row:
            out[k] = row[k]
    # 혹시 남은 키가 있으면 끝에
    for k, v in row.items():
        if k not in out:
            out[k] = v
    return out


# =========================
# CLI
# =========================
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate saved inference (COCO-style) against ground-truth annotations"
    )

    # Data and I/O
    group_data = parser.add_argument_group("Data and I/O")
    mx = group_data.add_mutually_exclusive_group(required=True)
    mx.add_argument("--set", type=str, choices=SET_REPLAYS.keys(),
                    help="Replay set (e.g., set_0, set_1, set_2).")
    mx.add_argument("--replays", nargs="+",
                    help="Explicit list of replay IDs (e.g., 1559 1628 2351 ...).")

    group_data.add_argument("--label-method", type=str,
                            default=config.LABEL_METHODS[0],
                            choices=config.LABEL_METHODS,
                            help="JSON label filename without extension.")

    group_data.add_argument("--gt-root", type=str,
                            default=os.path.join(os.getcwd(), "data"),
                            help="Ground-truth root (default: ./data). "
                                 "GT JSONs at <gt-root>/label/dst/[<gt_annotator>/]{replay}.rep/<label>.json")

    group_data.add_argument("--pred-root", type=str,
                            default=os.path.join(os.getcwd(), "predictions"),
                            help="Predictions root (default: ./predictions). "
                                 "pred-names are resolved relative to this root.")

    # Targets
    group_targets = parser.add_argument_group("Targets")
    group_targets.add_argument("--pred-names", nargs="+",
                               help="Prediction run names (can include subdirs) relative to --pred-root, "
                                    "e.g., 'vanilla/all_correct_win4_b16_20250812_062928'.")
    group_targets.add_argument("--model-number", type=int,
                               help="Model number to evaluate. If set, will read from '<pred-name>/model_<N>/'.")
    group_targets.add_argument("--gt-loo", action="store_true",
                               help="Leave-one-out using GT annotators (target=held-out GT). "
                                    "If set, predictions are ignored.")

    # Evaluation Options
    group_eval = parser.add_argument_group("Evaluation Options")
    group_eval.add_argument("--partial-length", type=float, default=1.0,
                            help="Fraction (0,1] of frames per replay to evaluate.")
    group_eval.add_argument(
        "--frame-select", nargs="+",
        choices=["first", "top1", "avg", "all"],
        default=["first"],
        help="Per-frame reduction: first/top1/avg or 'all' to evaluate all three."
    )
    group_eval.add_argument("--use-bbox-size", action="store_true",
                            help="Use scaled bbox [w,h] as viewport; otherwise fixed kernel size.")
    group_eval.add_argument("--ic-thresholds", type=float, nargs="+",
                            default=[0.0, 0.3, 0.5],
                            help="IC thresholds in [0,1] (e.g., 0 0.1 0.2 ... 0.9)")

    # Grid / Scaling
    group_grid = parser.add_argument_group("Grid / Scaling")
    group_grid.add_argument("--grid-width", type=int, default=128)
    group_grid.add_argument("--grid-height", type=int, default=128)
    group_grid.add_argument("--kernel-x", type=int, default=20)
    group_grid.add_argument("--kernel-y", type=int, default=12)
    group_grid.add_argument("--max-x", type=int, default=3456)
    group_grid.add_argument("--max-y", type=int, default=3720)

    # Output artifacts
    group_out = parser.add_argument_group("Output")
    group_out.add_argument("--out-dir", type=str,
                           default=os.path.join(os.getcwd(), "results"),
                           help="Directory to write JSON artifact (with metadata).")
    group_out.add_argument("--tag", type=str, default=None,
                           help="Optional tag to include in artifact filename.")
    group_out.add_argument("--dump-vpd", action="store_true",
                       help="Also dump per-frame VPD files (frame,vpx,vpy) for each frame-select.")
    group_out.add_argument("--vpd-dir", type=str,
                        default=os.path.join(os.getcwd(), "results", "vpd"),
                        help="Root directory to write VPD files (default: ./results/vpd).")
    group_out.add_argument("--vpd-clip", action="store_true",
                       help="Clip vpx/vpy to [0, max_x/max_y) using Grid.max_x/max_y.")

    # Logging
    group_log = parser.add_argument_group("Logging")
    group_log.add_argument("--log-level", type=str, default="log",
                           choices=["none", "log", "debug"],
                           help="Logger level (default: log).")

    args = parser.parse_args()

    # ---- Post-parse normalization ----
    if getattr(args, "set", None):
        args.replays = list(SET_REPLAYS[args.set])
    else:
        args.replays = list(dict.fromkeys(args.replays))  # dedupe

    # frame-select as list & expand all
    if isinstance(args.frame_select, str):
        args.frame_select = [args.frame_select]
    if "all" in args.frame_select:
        args.frame_select = ["first", "top1", "avg"]
    args.frame_select = list(dict.fromkeys(args.frame_select))  # dedupe, keep order

    # thresholds normalize/validate
    args.ic_thresholds = sorted(set(round(t, 4) for t in args.ic_thresholds))
    if any(t < 0.0 or t > 1.0 for t in args.ic_thresholds):
        parser.error("--ic-thresholds must be within [0, 1].")

    if not args.gt_loo and not args.pred_names:
        parser.error("--pred-names is required unless --gt-loo is set.")
    if not (0.0 < args.partial_length <= 1.0):
        parser.error("--partial-length must be in (0, 1].")

    return args


# =========================
# COCO loaders
# =========================
def _read_json(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def load_coco_track(
    root: Path, replay: str, label_method: str, frame_select: str
) -> pd.DataFrame:
    """
    <root>/<replay>.rep/<label>.json → DataFrame[frame,x,y,w,h,img_w,img_h,score]
    Frames aligned to 'images'; missing frames ffill/bfill.
    """
    jpath = root / f"{replay}.rep" / f"{label_method}.json"
    if not jpath.is_file():
        Logger.error(f"Missing JSON: {jpath}")
        raise FileNotFoundError(f"Missing JSON: {jpath}")
    coco = _read_json(jpath)

    images = pd.DataFrame(coco.get("images", []))
    anns   = pd.DataFrame(coco.get("annotations", []))

    if images.empty:
        Logger.warn(f"No images in {jpath}; returning empty track.")
        return pd.DataFrame(columns=['frame','x','y','w','h','img_w','img_h','score'])

    img_meta = images[['id','width','height']].rename(columns={'id':'frame','width':'img_w','height':'img_h'})

    if 'bbox' in anns.columns and not anns.empty:
        xywh = pd.DataFrame(anns['bbox'].tolist(), columns=['x','y','w','h'])
        anns = pd.concat([anns.drop(columns=['bbox']), xywh], axis=1)
    else:
        anns = pd.DataFrame(columns=['image_id','x','y','w','h','score'])

    if 'score' not in anns.columns:
        anns['score'] = np.nan

    anns = anns.rename(columns={'image_id': 'frame'})
    anns = anns.sort_values(['frame', 'score'], ascending=[True, False])

    if frame_select in ('first', 'top1'):
        # frame별 score 내림차순에서 첫 행만
        red = (
            anns.drop_duplicates('frame', keep='first')
                [['frame', 'x', 'y', 'w', 'h', 'score']]
                .reset_index(drop=True)
        )
    else:  # 'avg' — 중심점/크기 평균으로 대표 bbox 계산
        tmp = anns[['frame', 'x', 'y', 'w', 'h', 'score']].copy()
        tmp['cx'] = tmp['x'] + tmp['w'] / 2
        tmp['cy'] = tmp['y'] + tmp['h'] / 2
        gb = (
            tmp.groupby('frame', as_index=False)
               .agg({'cx': 'mean', 'cy': 'mean', 'w': 'mean', 'h': 'mean', 'score': 'mean'})
        )
        gb['x'] = gb['cx'] - gb['w'] / 2
        gb['y'] = gb['cy'] - gb['h'] / 2
        red = gb[['frame', 'x', 'y', 'w', 'h', 'score']]

    track = img_meta.merge(red, on='frame', how='left').sort_values('frame')
    track[['x','y','w','h','score']] = track[['x','y','w','h','score']].ffill().bfill()
    track['frame'] = track['frame'].astype(int)
    return track[['frame','x','y','w','h','img_w','img_h','score']]


def load_tracks_for_names(
    root: Path, names: List[str], replays: List[str], label_method: str, frame_select: str
) -> List[List[pd.DataFrame]]:
    """
    per-annotator 모드에서 사용: root/<name>/<replay>.rep/<label>.json
    결과 형태: tracks_per_name[name_idx][replay_idx] -> DataFrame
    """
    tracks_per_name: List[List[pd.DataFrame]] = []
    for name in names:
        base = root / name
        if not base.is_dir():
            Logger.error(f"Annotator dir not found: {base}")
            raise FileNotFoundError(f"Annotator dir not found: {base}")
        per_replay = [load_coco_track(base, rep, label_method, frame_select) for rep in replays]
        tracks_per_name.append(per_replay)
    return tracks_per_name


def load_tracks_flat(
    root: Path, replays: List[str], label_method: str, frame_select: str
) -> List[pd.DataFrame]:
    """
    flat 모드에서 사용: root/{replay}.rep/<label>.json
    결과 형태: tracks[replay_idx] -> DataFrame
    """
    return [load_coco_track(root, rep, label_method, frame_select) for rep in replays]


# =========================
# Frame unification
# =========================
def unify_frames(
    tracks_per_annotator: List[List[pd.DataFrame]], replays: List[str]
) -> List[List[pd.DataFrame]]:
    """
    각 replay별로 전체 annotator의 frame union으로 reindex → ffill → bfill
    """
    out: List[List[pd.DataFrame]] = []
    for per_replay in tracks_per_annotator:
        ann_out: List[pd.DataFrame] = []
        for r_idx, rep_df in enumerate(per_replay):
            frames_union = sorted(set().union(*[t[r_idx]['frame'].tolist() for t in tracks_per_annotator]))
            if not frames_union:
                ann_out.append(rep_df.copy())
                continue
            cur = rep_df.set_index('frame').reindex(frames_union).ffill().bfill().reset_index()
            ann_out.append(cur)
        out.append(ann_out)
    return out


# =========================
# Geometry helpers
# =========================
def _clip(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else hi if v > hi else v


def to_grid_rect(x: float, y: float, w: float, h: float,
                 img_w: Optional[float], img_h: Optional[float],
                 g: Grid, use_bbox_size: bool) -> tuple[int, int, int, int]:
    """
    bbox [x,y,w,h] → grid rect (gx,gy,gw,gh)
    - img_w/img_h 있으면 그것으로 스케일, 없으면 (g.max_x,g.max_y) 사용
    - use_bBox_size=False면 gw,gh는 (g.x_len,g.y_len) 고정
    """
    sx = (g.width  / float(img_w)) if (img_w and img_w > 0) else (g.width  / g.max_x)
    sy = (g.height / float(img_h)) if (img_h and img_h > 0) else (g.height / g.max_y)

    gx = int(round(x * sx))
    gy = int(round(y * sy))

    if use_bbox_size:
        gw = max(1, int(round(w * sx)))
        gh = max(1, int(round(h * sy)))
    else:
        gw, gh = g.x_len, g.y_len

    gx = _clip(gx, 0, g.width - 1)
    gy = _clip(gy, 0, g.height - 1)
    gx2 = _clip(gx + gw, 0, g.width)
    gy2 = _clip(gy + gh, 0, g.height)
    gw = max(1, gx2 - gx)
    gh = max(1, gy2 - gy)
    return gx, gy, gw, gh


# =========================
# Core evaluation
# =========================
def _mean(xs): return float(np.mean(xs)) if xs else 0.0


def _mean_streak(bools):
    # bool 리스트에서 True 연속 길이들의 평균
    if not bools:
        return 0.0
    streaks, cur = [], 0
    for v in bools:
        if v:
            cur += 1
        elif cur:
            streaks.append(cur)
            cur = 0
    if cur:
        streaks.append(cur)
    return float(np.mean(streaks)) if streaks else 0.0


def evaluate_intersection(
    replays: List[str],
    per_annotator_tracks: List[List[pd.DataFrame]],  # [target, ref1, ref2, ...]
    partial_length: float,
    g: Grid,
    use_bbox_size: bool,
    ic_thresholds: Optional[List[float]] = None,
) -> dict:
    ic_thresholds = sorted(set(ic_thresholds or [0.0, 0.3, 0.5]))

    # 누적 버퍼
    ovls, ious, dices, recalls = [], [], [], []
    ic_hits = {t: [] for t in ic_thresholds}          # 프레임 hit (True/False)

    # replay별 최소 길이 × partial-length
    min_lengths = []
    for r_idx, _ in enumerate(replays):
        lens = [len(a[r_idx]) for a in per_annotator_tracks]
        T = int(min(lens) * partial_length) if lens else 0
        min_lengths.append(max(0, T))

    for r_idx, _ in enumerate(replays):
        target = per_annotator_tracks[0][r_idx]
        refs   = [a[r_idx] for a in per_annotator_tracks[1:]]
        T = min_lengths[r_idx]
        if T <= 0 or target.empty or any(r.empty for r in refs):
            continue

        for t in range(T):
            canvas = np.zeros((g.height, g.width), dtype=np.int16)
            for r in refs:
                row = r.iloc[t]
                px, py, pw, ph = to_grid_rect(row.x, row.y, row.w, row.h, row.img_w, row.img_h, g, use_bbox_size)
                canvas[py:py+ph, px:px+pw] += 1

            union = (canvas > 0)

            rowt = target.iloc[t]
            rx, ry, rw, rh = to_grid_rect(rowt.x, rowt.y, rowt.w, rowt.h, rowt.img_w, rowt.img_h, g, use_bbox_size)
            patch = np.zeros_like(union, dtype=bool)
            patch[ry:ry+rh, rx:rx+rw] = True

            I  = np.logical_and(patch, union).sum()
            Ap = patch.sum()
            Ar = union.sum()
            U  = Ap + Ar - I

            ovl = (I / Ap) if Ap > 0 else 0.0               # precision 유사 (사용 지표)
            iou = (I / U)  if U  > 0 else 0.0               # 대칭형
            dice= (2*I/(Ap+Ar)) if (Ap+Ar) > 0 else 0.0     # 대칭형
            cr  = (I / Ar) if Ar > 0 else 0.0               # recall 유사

            ovls.append(ovl); ious.append(iou); dices.append(dice); recalls.append(cr)

            for thr in ic_thresholds:
                hit = (ovl >= thr) if thr > 0 else (ovl > 0.0)
                ic_hits[thr].append(hit)

    # 집계
    out = {}
    for thr in ic_thresholds:
        key = f"ic_at_{_thr_key(thr)}"
        out[key] = _mean(ic_hits[thr])

    out["ic_mean"]   = _mean(ovls)    # 프레임 평균
    out["iou_mean"]  = _mean(ious)
    out["dice_mean"] = _mean(dices)
    out["cr_mean"]   = _mean(recalls)

    return out


def evaluate_per_person(
    replays: List[str],
    per_annotator_tracks: List[List[pd.DataFrame]],  # [target, ref1, ref2, ...]
    partial_length: float,
    g: Grid,
    use_bbox_size: bool,
    ic_thresholds: Optional[List[float]] = None,
) -> List[dict]:
    """
    각 '참조자(ref)'를 한 사람으로 보고, target(=pred)과 1:1로 비교해
    사람별 metric을 산출한다. (모든 리플레이 평균)
    반환: [{person_idx: 0, ic_at_*, ic_mean, iou_mean, dice_mean, cr_mean}, ...]
    """
    ic_thresholds = sorted(set(ic_thresholds or [0.0, 0.3, 0.5]))

    n_person = max(0, len(per_annotator_tracks) - 1)
    results_per_person: List[dict] = []

    # replay별 최소 길이 × partial-length
    min_lengths = []
    for r_idx, _ in enumerate(replays):
        lens = [len(a[r_idx]) for a in per_annotator_tracks]
        T = int(min(lens) * partial_length) if lens else 0
        min_lengths.append(max(0, T))

    # 사람 루프 (ref_k: 1..N)
    for k in range(1, n_person + 1):
        ovls, ious, dices, recalls = [], [], [], []
        ic_hits = {t: [] for t in ic_thresholds}

        for r_idx, _ in enumerate(replays):
            target = per_annotator_tracks[0][r_idx]
            ref    = per_annotator_tracks[k][r_idx]
            T = min_lengths[r_idx]
            if T <= 0 or target.empty or ref.empty:
                continue

            for t in range(T):
                canvas = np.zeros((g.height, g.width), dtype=np.int16)
                row = ref.iloc[t]
                px, py, pw, ph = to_grid_rect(row.x, row.y, row.w, row.h, row.img_w, row.img_h, g, use_bbox_size)
                canvas[py:py+ph, px:px+pw] += 1
                union = (canvas > 0)

                rowt = target.iloc[t]
                rx, ry, rw, rh = to_grid_rect(rowt.x, rowt.y, rowt.w, rowt.h, rowt.img_w, rowt.img_h, g, use_bbox_size)
                patch = np.zeros_like(union, dtype=bool)
                patch[ry:ry+rh, rx:rx+rw] = True

                I  = np.logical_and(patch, union).sum()
                Ap = patch.sum()
                Ar = union.sum()
                U  = Ap + Ar - I

                ovl  = (I / Ap) if Ap > 0 else 0.0
                iou  = (I / U ) if U  > 0 else 0.0
                dice = (2*I/(Ap+Ar)) if (Ap+Ar) > 0 else 0.0
                cr   = (I / Ar) if Ar > 0 else 0.0

                ovls.append(ovl); ious.append(iou); dices.append(dice); recalls.append(cr)

                for thr in ic_thresholds:
                    hit = (ovl >= thr) if thr > 0 else (ovl > 0.0)
                    ic_hits[thr].append(hit)

        row = {"person_idx": k-1}
        for thr in ic_thresholds:
            row[f"ic_at_{_thr_key(thr)}"] = float(np.mean(ic_hits[thr])) if ic_hits[thr] else 0.0
        row["ic_mean"]   = float(np.mean(ovls))    if ovls else 0.0
        row["iou_mean"]  = float(np.mean(ious))    if ious else 0.0
        row["dice_mean"] = float(np.mean(dices))   if dices else 0.0
        row["cr_mean"]   = float(np.mean(recalls)) if recalls else 0.0

        results_per_person.append(row)

    return results_per_person


def evaluate_per_person_by_replay(
    replays: List[str],
    per_annotator_tracks: List[List[pd.DataFrame]],  # [target, ref1, ref2, ...]
    partial_length: float,
    g: Grid,
    use_bbox_size: bool,
    ic_thresholds: Optional[List[float]] = None,
) -> Dict[str, List[dict]]:
    """
    리플레이별로, 각 사람(ref)과 target(=pred)을 1:1 비교한 메트릭 목록을 반환.
    반환:
      {
        "<replay_id>": [
          {"person_idx": 0, "ic_at_0_3": ..., "ic_mean": ..., "iou_mean": ..., "dice_mean": ..., "cr_mean": ...},
          ...
        ],
        ...
      }
    """
    ic_thresholds = sorted(set(ic_thresholds or [0.0, 0.3, 0.5]))
    n_person = max(0, len(per_annotator_tracks) - 1)

    # replay별 최소 길이 × partial-length
    min_lengths = []
    for r_idx, _ in enumerate(replays):
        lens = [len(a[r_idx]) for a in per_annotator_tracks]
        T = int(min(lens) * partial_length) if lens else 0
        min_lengths.append(max(0, T))

    out_by_rep: Dict[str, List[dict]] = {}

    for r_idx, rep_id in enumerate(replays):
        T = min_lengths[r_idx]
        if T <= 0:
            out_by_rep[rep_id] = []
            continue

        target = per_annotator_tracks[0][r_idx]
        if target.empty:
            out_by_rep[rep_id] = []
            continue

        per_person_rows: List[dict] = []

        for k in range(1, n_person + 1):
            ref = per_annotator_tracks[k][r_idx]
            if ref.empty:
                continue

            ovls, ious, dices, recalls = [], [], [], []
            ic_hits = {t: [] for t in ic_thresholds}

            for t in range(T):
                canvas = np.zeros((g.height, g.width), dtype=np.int16)
                row = ref.iloc[t]
                px, py, pw, ph = to_grid_rect(row.x, row.y, row.w, row.h, row.img_w, row.img_h, g, use_bbox_size)
                canvas[py:py+ph, px:px+pw] += 1
                union = (canvas > 0)

                rowt = target.iloc[t]
                rx, ry, rw, rh = to_grid_rect(rowt.x, rowt.y, rowt.w, rowt.h, rowt.img_w, rowt.img_h, g, use_bbox_size)
                patch = np.zeros_like(union, dtype=bool)
                patch[ry:ry+rh, rx:rx+rw] = True

                I  = np.logical_and(patch, union).sum()
                Ap = patch.sum()
                Ar = union.sum()
                U  = Ap + Ar - I

                ovl  = (I / Ap) if Ap > 0 else 0.0
                iou  = (I / U ) if U  > 0 else 0.0
                dice = (2*I/(Ap+Ar)) if (Ap+Ar) > 0 else 0.0
                cr   = (I / Ar) if Ar > 0 else 0.0

                ovls.append(ovl); ious.append(iou); dices.append(dice); recalls.append(cr)
                for thr in ic_thresholds:
                    hit = (ovl >= thr) if thr > 0 else (ovl > 0.0)
                    ic_hits[thr].append(hit)

            row = {"person_idx": k-1}
            for thr in ic_thresholds:
                row[f"ic_at_{_thr_key(thr)}"] = float(np.mean(ic_hits[thr])) if ic_hits[thr] else 0.0
            row["ic_mean"]   = float(np.mean(ovls))    if ovls else 0.0
            row["iou_mean"]  = float(np.mean(ious))    if ious else 0.0
            row["dice_mean"] = float(np.mean(dices))   if dices else 0.0
            row["cr_mean"]   = float(np.mean(recalls)) if recalls else 0.0

            per_person_rows.append(row)

        out_by_rep[rep_id] = per_person_rows

    return out_by_rep


def detect_gt_structure(gt_dst_dir: Path) -> str:
    """
    Returns 'flat' if <dst> contains {rep}.rep dirs directly,
            'per_annotator' if it contains annotator subdirs.
    """
    if not gt_dst_dir.is_dir():
        Logger.error(f"GT directory not found: {gt_dst_dir}")
        raise FileNotFoundError(f"GT directory not found: {gt_dst_dir}")

    children = [p for p in gt_dst_dir.iterdir() if p.is_dir()]
    if any(c.name.endswith(".rep") for c in children):
        return "flat"
    if any(not c.name.endswith(".rep") for c in children):
        return "per_annotator"
    return "flat"


def _summary_ic_line(stats: dict, thrs: List[float]) -> str:
    left = "IC@" + "/".join(str(t).rstrip("0").rstrip(".") for t in thrs)
    vals = []
    for t in thrs:
        k = f"ic_at_{_thr_key(t)}"
        vals.append(f"{stats.get(k, 0.0):.2f}")
    right = "/".join(vals)
    return f"{left} = {right}"


# =========================
# Orchestration
# =========================
def run_evaluate(args):
    g = Grid.from_args(args)

    timestamp = _now_utc_str()
    replays = args.replays
    gt_base = Path(args.gt_root)            # e.g., /workspace/data
    gt_dst  = gt_base / "label" / "dst"     # /workspace/data/label/dst
    pred_root = Path(args.pred_root)        # e.g., /workspace/predictions

    _assert_dir(gt_base, "GT base")
    _assert_dir(gt_dst,  "GT dst (expected <gt-root>/label/dst)")
    _assert_dir(pred_root, "Predictions root")

    # 예측 런 디렉터리 확인 (model_number 반영)
    if not args.gt_loo:
        missing = []
        for name in args.pred_names:
            base = pred_root / name
            if args.model_number is not None:
                base = base / f"model_{args.model_number:03d}"
            if not base.is_dir():
                missing.append(str(base))
        if missing:
            Logger.error("Prediction run directory(ies) not found:")
            for m in missing:
                Logger.error(f"  - {m}")
            raise FileNotFoundError("One or more prediction run directories are missing.")

    gt_mode = detect_gt_structure(gt_dst)

    Logger.log("=== Evaluation (predictions vs Ground Truth) ===")
    Logger.log(f"Replays = {replays}")
    Logger.log(f"Label   = {args.label_method} | frame-selects = {args.frame_select} | partial = {args.partial_length}")
    Logger.log(f"IC thresholds = {args.ic_thresholds}")
    Logger.log(f"GT base = {gt_base}  (mode: {gt_mode})")
    if args.model_number is not None:
        Logger.log(f"Pred root = {pred_root} (using model_{args.model_number:03d} subfolders)")
    else:
        Logger.log(f"Pred root = {pred_root}")
    Logger.log(f"Pred names = {args.pred_names if args.pred_names else 'N/A (GT LOO)'}")
    Logger.log("-" * 60)

    rows = []
    per_person_by_fs: Dict[str, Dict[str, List[dict]]] = {}
    per_person_by_replay: Dict[str, Dict[str, Dict[str, List[dict]]]] = {}

    # FS 루프
    for fs in args.frame_select:
        # --- GT 로딩 (refs) ---
        if gt_mode == "flat":
            gt_tracks_flat = load_tracks_flat(gt_dst, replays, args.label_method, fs)
            gt_refs = [gt_tracks_flat]  # wrap as a single "annotator"
        else:
            gt_annotators = SET_GT_ANNOTATORS[args.set] if getattr(args, "set", None) else []
            gt_refs = load_tracks_for_names(gt_dst, gt_annotators, replays,
                                            args.label_method, fs)
            
        if args.gt_loo:
            Logger.error("GT LOO mode not implemented in multi-FS loop (set --pred-names instead).")
            raise RuntimeError("GT LOO mode currently not supported when looping multiple frame-selects.")
        else:
            for pred in args.pred_names:
                # pred 로더용 경로(모델 번호 포함해 읽음)
                pred_name_for_loader = f"{pred}/model_{args.model_number:03d}" if args.model_number is not None else pred
                pred_tracks = load_tracks_for_names(
                    pred_root, [pred_name_for_loader], replays, args.label_method, fs
                )[0]

                per_annotator_tracks = [pred_tracks] + gt_refs
                per_annotator_tracks = unify_frames(per_annotator_tracks, replays)
                stats = evaluate_intersection(
                    replays, per_annotator_tracks, args.partial_length, g, args.use_bbox_size,
                    ic_thresholds=args.ic_thresholds
                )
                per_person_rows = evaluate_per_person(
                    replays, per_annotator_tracks, args.partial_length, g, args.use_bbox_size,
                    ic_thresholds=args.ic_thresholds
                )
                per_person_by_fs.setdefault(pred, {})[fs] = per_person_rows

                # 사람×리플레이
                pp_by_rep = evaluate_per_person_by_replay(
                    replays, per_annotator_tracks, args.partial_length, g, args.use_bbox_size,
                    ic_thresholds=args.ic_thresholds
                )
                per_person_by_replay.setdefault(pred, {})[fs] = pp_by_rep

                rows.append({'target': pred,
                             'frame_select': fs,
                             'per_person_ic_mean_avg': float(np.mean([r['ic_mean'] for r in per_person_rows])) if per_person_rows else 0.0,
                             **stats})

                # Per-person 테이블 로그
                if per_person_rows:
                    df_pp = pd.DataFrame(per_person_rows)
                    rename_map = {f"ic_at_{_thr_key(t)}": f"IC@{t:g}" for t in args.ic_thresholds}
                    rename_map.update({
                        "ic_mean": "IC_mean", "iou_mean": "IoU_mean",
                        "dice_mean": "Dice_mean", "cr_mean": "CR_mean",
                        "person_idx": "Person"
                    })
                    Logger.log(f"[PRED {pred_name_for_loader} | FS={fs}] Per-Person Metrics")
                    Logger.log(df_pp.rename(columns=rename_map).to_string(index=False))

                # Per-replay × person (IC_mean) 요약 로그
                try:
                    max_person = 1 + max((r["person_idx"] for r in per_person_rows), default=-1)
                except Exception:
                    max_person = 0
                if max_person > 0 and pp_by_rep:
                    persons = list(range(1, max_person+1))
                    Logger.log(f"[PRED {pred_name_for_loader} | FS={fs}] Per-Replay IC_mean (rows=replay, cols=person)")
                    Logger.log("\t" + "\t".join(map(str, persons)))
                    for rid in replays:
                        row = pp_by_rep.get(rid, [])
                        pmap = { (r["person_idx"]+1): r.get("ic_mean", 0.0) for r in row }
                        vals = [f"{pmap.get(p, 0.0):.4f}" for p in persons]
                        Logger.log(f"{rid}\t" + "\t".join(vals))

                # --- 여기서 pred별 출력 루트 생성 ---
                pred_out_root = Path(args.out_dir) / pred
                if args.model_number is not None:
                    pred_out_root = pred_out_root / f"model_{args.model_number:03d}"
                pred_out_root = pred_out_root / timestamp
                pred_out_root.mkdir(parents=True, exist_ok=True)

                if args.dump_vpd:
                    clip_max = (g.max_x, g.max_y) if args.vpd_clip else None
                    vpd_root = pred_out_root / "vpd"
                    _dump_vpd_for_tracks(
                        frame_select=fs,
                        replays=replays,
                        tracks=pred_tracks,
                        vpd_root=vpd_root,
                        scale=32,
                        clip_max=clip_max
                    )

                Logger.log(f"[PRED {pred_name_for_loader} | FS={fs}] " + _summary_ic_line(stats, args.ic_thresholds))

    # === Final aggregation & JSON artifact ===
    if not rows:
        Logger.warn("No results.")
        return

    df_out = pd.DataFrame(rows)

    # 전체 평균
    metric_cols = [c for c in df_out.columns if c not in ("target", "frame_select")]
    mean_overall = {m: float(df_out[m].mean()) for m in metric_cols if m in df_out}

    df_out = pd.DataFrame(rows)

    # 보기 좋은 테이블 로그 (기존 출력 유지)
    for fs in args.frame_select:
        sub = df_out[df_out["frame_select"] == fs].copy()
        rename_map = {f"ic_at_{_thr_key(t)}": f"IC@{t:g}" for t in args.ic_thresholds}
        rename_map.update({
            "ic_mean": "IC_mean",
            "iou_mean": "IoU_mean",
            "dice_mean": "Dice_mean",
            "cr_mean": "CR_mean"
        })
        Logger.log(f"--- FrameSelect = {fs} ---")
        Logger.log(sub.rename(columns=rename_map).to_string(index=False))

    # 공통 메타 (pred별 result.json 안에 동일하게 포함)
    gt_mode = detect_gt_structure(Path(args.gt_root) / "label" / "dst")
    meta_common = {
        "timestamp_utc": timestamp,
        "gt_root": str(Path(args.gt_root).resolve()),
        "pred_root": str(Path(args.pred_root).resolve()),
        "gt_mode": gt_mode,
        "replays": args.replays,
        "label_method": args.label_method,
        "frame_selects": args.frame_select,
        "partial_length": args.partial_length,
        "use_bbox_size": bool(args.use_bbox_size),
        "grid": {
            "width": g.width, "height": g.height,
            "kernel_x": g.x_len, "kernel_y": g.y_len,
            "max_x": g.max_x, "max_y": g.max_y,
        },
        "model_number": args.model_number,
        "gt_loo": bool(args.gt_loo),
        "tag": args.tag,
        "script": "evaluate.py",
        "version": "1.2",
        "ic_thresholds": args.ic_thresholds,
    }
    
    # --- pred 별로 result.json 쓰기 ---
    for pred in (args.pred_names or []):
        pred_rows = [r for r in rows if r.get("target") == pred]
        if not pred_rows:
            continue
        pred_df = pd.DataFrame(pred_rows)

        metric_cols = [c for c in pred_df.columns if c not in ("target", "frame_select")]
        mean_overall = {m: float(pred_df[m].mean()) for m in metric_cols if m in pred_df}

        mean_by_fs = {}
        for fs in args.frame_select:
            sub = pred_df[pred_df["frame_select"] == fs]
            mean_by_fs[fs] = {m: float(sub[m].mean()) for m in metric_cols if m in sub}

        rows_ordered = [_order_row(r, args.ic_thresholds) for r in pred_rows]

        def _order_metric_dict(d: dict, ic_thresholds) -> OrderedDict:
            has_baseline = any(k.startswith("streak_baseline_at_") for k in d)
            has_ratio    = any(k.startswith("streak_ratio_at_") for k in d)
            keys         = _ordered_metric_keys(ic_thresholds, has_baseline, has_ratio)
            out = OrderedDict()
            for k in ("target","frame_select"):
                if k in d: out[k] = d[k]
            for k in keys:
                if k in d: out[k] = d[k]
            for k, v in d.items():
                if k not in out: out[k] = v
            return out

        payload = {
            "meta": {**meta_common, "pred_names": [pred]},
            "results": {
                "by_target": rows_ordered,
                "mean_overall": _order_metric_dict(mean_overall, args.ic_thresholds),
                "mean_by_frame_select": {fs: _order_metric_dict(v, args.ic_thresholds) for fs, v in mean_by_fs.items()},
                "per_person_by_frame_select": {
                     fs: [
                         _order_metric_dict({"frame_select": fs, **row}, args.ic_thresholds)
                         for row in per_person_by_fs.get(pred, {}).get(fs, [])
                     ] for fs in args.frame_select
                 },
                "per_person_by_replay": {
                    fs: {
                        rep: [
                            _order_metric_dict({"frame_select": fs, **row}, args.ic_thresholds)
                            for row in per_person_by_replay.get(pred, {}).get(fs, {}).get(rep, [])
                        ] for rep in meta_common["replays"]
                    } for fs in args.frame_select
                },
            },
        }

        pred_out_root = Path(args.out_dir) / pred
        if args.model_number is not None:
            pred_out_root = pred_out_root / f"model_{args.model_number:03d}"
        pred_out_root = pred_out_root / timestamp
        pred_out_root.mkdir(parents=True, exist_ok=True)

        artifact_json = pred_out_root / "result.json"
        with artifact_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        Logger.info(f"Wrote artifact JSON -> {artifact_json}")


# =========================
# Entrypoint
# =========================
def main():
    args = parse_arguments()
    Logger.set_level(args.log_level)
    run_evaluate(args)


if __name__ == '__main__':
    main()
