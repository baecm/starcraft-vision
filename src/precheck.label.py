import argparse
import json
import os
from collections import Counter
from typing import Any, Dict, List, Tuple, Optional

from tqdm import tqdm


def parse_int_list(s: str) -> List[int]:
    out = []
    for p in s.replace(",", " ").split():
        if p.strip():
            out.append(int(p))
    return out


def append_jsonl(path: str, rec: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_finite_num(x) -> bool:
    try:
        return x == x and x not in (float("inf"), float("-inf"))
    except Exception:
        return False


def validate_coco_json(
    data: Dict[str, Any],
    rid: str,
    path: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []

    images = data.get("images", [])
    anns = data.get("annotations", [])
    cats = data.get("categories", [])

    if not isinstance(images, list):
        issues.append({"kind": "bad_schema", "rid": rid, "path": path, "where": "images_not_list"})
        images = []
    if not isinstance(anns, list):
        issues.append({"kind": "bad_schema", "rid": rid, "path": path, "where": "annotations_not_list"})
        anns = []
    if cats is not None and not isinstance(cats, list):
        issues.append({"kind": "bad_schema", "rid": rid, "path": path, "where": "categories_not_list"})
        cats = []

    img_wh: Dict[int, Tuple[Optional[int], Optional[int]]] = {}
    dup_img_id = 0
    seen_img = set()

    for im in images:
        try:
            iid = int(im.get("id"))
        except Exception:
            issues.append({"kind": "bad_image", "rid": rid, "path": path, "where": "image_missing_id", "image": im})
            continue

        if iid in seen_img:
            dup_img_id += 1
        seen_img.add(iid)

        w = im.get("width", None)
        h = im.get("height", None)
        try:
            w = int(w) if w is not None else None
            h = int(h) if h is not None else None
        except Exception:
            w, h = None, None
        img_wh[iid] = (w, h)

    ann_per_image = Counter()
    for ann in anns:
        try:
            iid = int(ann.get("image_id"))
        except Exception:
            continue
        ann_per_image[iid] += 1

    mismatch = 0
    for iid, (w, h) in img_wh.items():
        c = ann_per_image.get(iid, 0)
        if c != 5:
            mismatch += 1
            issues.append({
                "kind": "ann_count_mismatch",
                "rid": rid,
                "path": path,
                "image_id": iid,
                "ann_count": c,
                "expected": 5,
            })

    bad_bbox = 0
    out_of_bounds = 0
    missing_img = 0
    bad_cat = 0
    nonfinite_bbox = 0

    for ann in anns:
        try:
            iid = int(ann.get("image_id"))
        except Exception:
            issues.append({"kind": "bad_ann", "rid": rid, "path": path, "where": "ann_missing_image_id", "ann": ann})
            continue

        if iid not in img_wh:
            missing_img += 1
            issues.append({"kind": "missing_image_ref", "rid": rid, "path": path, "image_id": iid, "ann_id": ann.get("id")})
            w = h = None
        else:
            w, h = img_wh[iid]

        cid = ann.get("category_id", None)
        try:
            cid_int = int(cid) if cid is not None else None
        except Exception:
            cid_int = None
        if cid_int is None or cid_int < 1:
            bad_cat += 1
            issues.append({"kind": "bad_category_id", "rid": rid, "path": path, "image_id": iid, "ann_id": ann.get("id"), "category_id": cid})

        bbox = ann.get("bbox", None)
        if bbox is None or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            bad_bbox += 1
            issues.append({"kind": "bad_bbox", "rid": rid, "path": path, "image_id": iid, "ann_id": ann.get("id"), "bbox": bbox})
            continue

        x, y, bw, bh = bbox
        if not (is_finite_num(x) and is_finite_num(y) and is_finite_num(bw) and is_finite_num(bh)):
            nonfinite_bbox += 1
            issues.append({"kind": "nonfinite_bbox", "rid": rid, "path": path, "image_id": iid, "ann_id": ann.get("id"), "bbox": bbox})
            continue

        try:
            bw_f = float(bw)
            bh_f = float(bh)
            x_f = float(x)
            y_f = float(y)
        except Exception:
            nonfinite_bbox += 1
            issues.append({"kind": "bad_bbox_cast", "rid": rid, "path": path, "image_id": iid, "ann_id": ann.get("id"), "bbox": bbox})
            continue

        if bw_f <= 0 or bh_f <= 0:
            bad_bbox += 1
            issues.append({"kind": "nonpositive_bbox", "rid": rid, "path": path, "image_id": iid, "ann_id": ann.get("id"), "bbox": bbox})
            continue

        if w is not None and h is not None:
            x2 = x_f + bw_f
            y2 = y_f + bh_f
            if x_f < 0 or y_f < 0 or x2 > w or y2 > h:
                out_of_bounds += 1
                issues.append({
                    "kind": "bbox_out_of_bounds",
                    "rid": rid,
                    "path": path,
                    "image_id": iid,
                    "ann_id": ann.get("id"),
                    "bbox": bbox,
                    "img_wh": [w, h],
                })

    stats = {
        "rid": rid,
        "path": path,
        "num_images": len(images),
        "num_annotations": len(anns),
        "dup_image_ids": dup_img_id,
        "missing_image_ref": missing_img,
        "bad_category_id": bad_cat,
        "bad_bbox": bad_bbox,
        "nonfinite_bbox": nonfinite_bbox,
        "bbox_out_of_bounds": out_of_bounds,
        "ann_count_mismatch": mismatch,
    }
    return issues, stats


def main():
    ap = argparse.ArgumentParser("Audit COCO label JSONs by replay list (tqdm)")
    ap.add_argument("--root_dir", required=True, help="e.g., /mnt/nas/.../data/label/dst")
    ap.add_argument("--label_method", required=True, help="e.g., all_correct")
    ap.add_argument("--replays", required=True, help='e.g., "275,1725,3613"')
    ap.add_argument("--out_dir", default="", help="default: <root_dir>/audit_labels")

    # tqdm / logging
    ap.add_argument("--verbose", action="store_true", help="print a line per replay using tqdm.write()")
    ap.add_argument("--tqdm_update_every", type=int, default=1,
                    help="how often to update tqdm postfix (1=every replay)")

    args = ap.parse_args()

    root_dir = args.root_dir
    label_method = args.label_method
    replays = [str(r) for r in parse_int_list(args.replays)]

    out_dir = args.out_dir or os.path.join(root_dir, "audit_labels")
    os.makedirs(out_dir, exist_ok=True)

    issues_path = os.path.join(out_dir, f"issues_{label_method}.jsonl")
    stats_path = os.path.join(out_dir, f"stats_{label_method}.json")
    missing_path = os.path.join(out_dir, f"missing_{label_method}.json")

    all_stats = []
    kind_ctr = Counter()
    missing_files = []

    total_images = 0
    total_anns = 0
    total_issue_records = 0

    pbar = tqdm(replays, desc=f"audit[{label_method}]", unit="replay", dynamic_ncols=True)
    for idx, rid in enumerate(pbar, 1):
        json_path = os.path.join(root_dir, f"{rid}.rep", f"{label_method}.json")
        if not os.path.isfile(json_path):
            missing_files.append({"rid": rid, "path": json_path})
            if args.verbose:
                tqdm.write(f"[Missing] rid={rid} -> {json_path}")
            continue

        try:
            data = load_json(json_path)
        except Exception as e:
            rec = {"kind": "json_load_error", "rid": rid, "path": json_path, "error": repr(e)}
            append_jsonl(issues_path, rec)
            kind_ctr[rec["kind"]] += 1
            total_issue_records += 1
            if args.verbose:
                tqdm.write(f"[LoadError] rid={rid} err={repr(e)}")
            continue

        issues, stats = validate_coco_json(data, rid=rid, path=json_path)
        all_stats.append(stats)

        total_images += int(stats.get("num_images", 0))
        total_anns += int(stats.get("num_annotations", 0))

        for it in issues:
            append_jsonl(issues_path, it)
            kind_ctr[it["kind"]] += 1
            total_issue_records += 1

        if args.verbose:
            tqdm.write(
                f"[Replay] rid={rid} images={stats['num_images']} anns={stats['num_annotations']} "
                f"issues={len(issues)} bad_bbox={stats['bad_bbox']} nonfinite_bbox={stats['nonfinite_bbox']} "
                f"oob={stats['bbox_out_of_bounds']} missing_img_ref={stats['missing_image_ref']} "
                f"dup_img_ids={stats['dup_image_ids']} ann_count_mismatch={stats['ann_count_mismatch']}"
            )

        # tqdm postfix는 너무 자주 바꾸면 느릴 수 있어서 옵션으로 조절
        if args.tqdm_update_every > 0 and (idx % args.tqdm_update_every == 0):
            pbar.set_postfix({
                "missing": len(missing_files),
                "issues": total_issue_records,
                "kinds": len(kind_ctr),
                "imgs": total_images,
                "anns": total_anns,
            })

    summary = {
        "root_dir": root_dir,
        "label_method": label_method,
        "replays": replays,
        "missing_files": missing_files,
        "issue_kind_counts": dict(kind_ctr),
        "stats": all_stats,
        "totals": {
            "num_replays": len(replays),
            "missing_files": len(missing_files),
            "total_images": total_images,
            "total_annotations": total_anns,
            "total_issue_records": total_issue_records,
        },
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(missing_path, "w", encoding="utf-8") as f:
        json.dump(missing_files, f, ensure_ascii=False, indent=2)

    print(f"[Done] out_dir={out_dir}")
    print(f"[Done] wrote issues : {issues_path}")
    print(f"[Done] wrote stats  : {stats_path}")
    print(f"[Done] wrote missing: {missing_path}")
    print(f"[Done] issue kinds : {dict(kind_ctr)}")
    if missing_files:
        print(f"[Done] missing files: {len(missing_files)}")


if __name__ == "__main__":
    main()
