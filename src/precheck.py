import argparse
import json
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def parse_int_list(s: str) -> List[int]:
    out = []
    for p in s.replace(",", " ").split():
        if p.strip():
            out.append(int(p))
    return out


def natural_key(name: str):
    base = os.path.basename(name)
    m = re.match(r"(\d+)\.npy$", base)
    if m:
        return int(m.group(1))
    return base


def append_jsonl(path: str, rec: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def list_replays(root_dir: str) -> List[str]:
    rids = []
    for name in os.listdir(root_dir):
        if name.endswith(".rep"):
            rids.append(name[:-4])
    rids.sort(key=lambda x: int(x) if x.isdigit() else x)
    return rids


# -----------------------------
# Worker (must be top-level)
# -----------------------------
def worker_check_one(args) -> Dict[str, Any]:
    """
    args: (rid, path, mmap, max_abs_warn, expect_shape)
    return:
      - {"ok": True, ... minimal stats ...}
      - {"ok": False, "issues": [..], ...}
    """
    rid, path, mmap, max_abs_warn, expect_shape = args
    issues = []

    try:
        arr = np.load(path, mmap_mode="r" if mmap else None)
    except Exception as e:
        issues.append({"kind": "load_error", "rid": rid, "path": path, "error": repr(e)})
        return {"ok": False, "rid": rid, "path": path, "issues": issues}

    shp = tuple(arr.shape)
    dt = str(arr.dtype)

    if expect_shape is not None and shp != expect_shape:
        issues.append({"kind": "bad_shape", "rid": rid, "path": path, "shape": list(shp), "expect": list(expect_shape)})

    # nonfinite
    finite = np.isfinite(arr)
    if not finite.all():
        nan_cnt = int(np.isnan(arr).sum())
        inf_cnt = int(np.isinf(arr).sum())
        issues.append({
            "kind": "nonfinite_input",
            "rid": rid,
            "path": path,
            "shape": list(shp),
            "dtype": dt,
            "nan": nan_cnt,
            "inf": inf_cnt,
        })

    # max_abs
    # nan은 무시, inf 있으면 max_abs=inf
    try:
        ma = float(np.nanmax(np.abs(arr)))
    except Exception:
        ma = None

    if max_abs_warn and ma is not None and ma > max_abs_warn:
        issues.append({
            "kind": "max_abs_exceed",
            "rid": rid,
            "path": path,
            "max_abs": ma,
            "threshold": float(max_abs_warn),
            "shape": list(shp),
            "dtype": dt,
        })

    return {
        "ok": len(issues) == 0,
        "rid": rid,
        "path": path,
        "shape": list(shp),
        "dtype": dt,
        "max_abs": ma,
        "issues": issues,
    }


def main():
    ap = argparse.ArgumentParser("Audit input npy (multiprocessing)")
    ap.add_argument("--root_dir", required=True)
    ap.add_argument("--replays", default="all")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--mmap", action="store_true")
    ap.add_argument("--max_abs_warn", type=float, default=0.0)
    ap.add_argument("--expect_shape", default="")
    ap.add_argument("--workers", type=int, default=0, help="0=auto (min(8,cpu_count))")
    ap.add_argument("--chunksize", type=int, default=64)
    args = ap.parse_args()

    root_dir = args.root_dir
    out_dir = args.out_dir or os.path.join(root_dir, "audit_inputs_mp")
    os.makedirs(out_dir, exist_ok=True)

    issues_path = os.path.join(out_dir, "issues_input.jsonl")
    summary_path = os.path.join(out_dir, "summary_input.json")

    # replay list
    if args.replays.strip().lower() == "all":
        rids = list_replays(root_dir)
    else:
        rids = [str(x) for x in parse_int_list(args.replays)]

    # expected shape
    expect_shape = None
    if args.expect_shape.strip():
        parts = [int(x) for x in args.expect_shape.replace(" ", "").split(",")]
        if len(parts) != 3:
            raise ValueError("--expect_shape must be like 'C,H,W'")
        expect_shape = tuple(parts)

    # build file list
    tasks = []
    for rid in rids:
        rep_dir = os.path.join(root_dir, f"{rid}.rep")
        if not os.path.isdir(rep_dir):
            append_jsonl(issues_path, {"kind": "missing_replay_dir", "rid": rid, "path": rep_dir})
            continue
        fps = [fn for fn in os.listdir(rep_dir) if fn.endswith(".npy")]
        fps.sort(key=natural_key)
        for fn in fps:
            path = os.path.join(rep_dir, fn)
            tasks.append((rid, path, args.mmap, args.max_abs_warn, expect_shape))

    # workers
    if args.workers <= 0:
        workers = min(8, cpu_count())
    else:
        workers = args.workers

    kind_ctr = Counter()
    per_rid = defaultdict(lambda: {"files": 0, "nonfinite": 0, "maxabs": 0, "load_error": 0, "bad_shape": 0})
    top_maxabs: List[Tuple[float, str]] = []

    def push_top(ma: float, path: str, k: int = 50):
        nonlocal top_maxabs
        top_maxabs.append((ma, path))
        top_maxabs.sort(key=lambda x: x[0], reverse=True)
        if len(top_maxabs) > k:
            top_maxabs = top_maxabs[:k]

    with Pool(processes=workers) as pool:
        it = pool.imap_unordered(worker_check_one, tasks, chunksize=args.chunksize)
        for res in tqdm(it, total=len(tasks), desc=f"audit_input_mp[w={workers}]", unit="file", dynamic_ncols=True):
            rid = res["rid"]
            per_rid[rid]["files"] += 1

            ma = res.get("max_abs", None)
            if isinstance(ma, (int, float)) and ma is not None:
                push_top(float(ma), res["path"])

            for issue in res.get("issues", []):
                append_jsonl(issues_path, issue)
                kind_ctr[issue["kind"]] += 1
                if issue["kind"] == "nonfinite_input":
                    per_rid[rid]["nonfinite"] += 1
                elif issue["kind"] == "max_abs_exceed":
                    per_rid[rid]["maxabs"] += 1
                elif issue["kind"] == "load_error":
                    per_rid[rid]["load_error"] += 1
                elif issue["kind"] == "bad_shape":
                    per_rid[rid]["bad_shape"] += 1

    summary = {
        "root_dir": root_dir,
        "replays": rids,
        "workers": workers,
        "chunksize": args.chunksize,
        "mmap": bool(args.mmap),
        "max_abs_warn": float(args.max_abs_warn),
        "expect_shape": list(expect_shape) if expect_shape is not None else None,
        "issue_kind_counts": dict(kind_ctr),
        "per_replay": {rid: dict(st) for rid, st in per_rid.items()},
        "top_max_abs": [{"max_abs": float(ma), "path": path} for ma, path in top_maxabs],
        "outputs": {"issues_jsonl": issues_path, "summary_json": summary_path},
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[Done] out_dir={out_dir}")
    print(f"[Done] issues : {issues_path}")
    print(f"[Done] summary: {summary_path}")
    print(f"[Done] issue kinds: {dict(kind_ctr)}")


if __name__ == "__main__":
    main()
