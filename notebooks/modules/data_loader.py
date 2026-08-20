import os
import glob
import yaml
import json
import pandas as pd
from pathlib import Path


def find_experiments(predictions_dir="/workspace/predictions", model_name_filter=None):
    """
    YAML 없이 predictions/ 디렉터리를 동적으로 자동 스캔하여
    존재하는 모든 실험 정보(아키텍처, 윈도우 크기, mode, epoch 등)를 파싱합니다.
    """
    if not os.path.exists(predictions_dir):
        alt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "predictions"))
        if os.path.exists(alt_path):
            predictions_dir = alt_path
        else:
            return []

    experiments = []
    for item in sorted(os.listdir(predictions_dir)):
        item_path = os.path.join(predictions_dir, item)
        if not os.path.isdir(item_path):
            continue

        if model_name_filter and not any(f.lower() in item.lower() for f in (model_name_filter if isinstance(model_name_filter, (list, tuple, set)) else [model_name_filter])):
            continue

        arch = "maskrcnn" if "maskrcnn" in item else ("centernet" if "centernet" in item else ("deformable_detr" if "deformable" in item or "detr" in item else "unknown"))
        win = 4 if "win4" in item else (1 if "win1" in item else 1)
        mode = "kbrs" if "kbrs" in item else "vanilla"

        epoch_dirs = [d for d in os.listdir(item_path) if d.startswith("model_") and os.path.isdir(os.path.join(item_path, d))]
        epochs = sorted([int(d.replace("model_", "")) for d in epoch_dirs if d.replace("model_", "").isdigit()])

        experiments.append({
            "model_name": item,
            "path": item_path,
            "arch": arch,
            "window_size": win,
            "mode": mode,
            "epochs": epochs,
            "glob_pattern": f"*{item}*",
        })

    return experiments


def load_config(config_path="config.yaml"):
    """
    YAML 설정 파일을 불러옵니다 (파일이 없으면 predictions/ 디렉터리 동적 스캔으로 자동 대체).
    """
    candidate_paths = [
        config_path,
        os.path.join(os.path.dirname(__file__), "..", config_path),
        os.path.join(os.getcwd(), config_path),
        os.path.join(os.getcwd(), "..", config_path),
        os.path.join("/workspace/notebooks", config_path),
    ]
    target_path = None
    for p in candidate_paths:
        if p and os.path.isfile(p):
            target_path = p
            break

    if target_path:
        with open(target_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # Fallback to dynamic Auto-Discovery dictionary if YAML file is absent
    discovered = find_experiments()
    exp_dict = {}
    for exp in discovered:
        arch = exp["arch"]
        win_key = f"win{exp['window_size']}"
        mode_key = exp["mode"]
        exp_dict.setdefault(arch, {}).setdefault(win_key, {})[mode_key] = exp["glob_pattern"]

    return {
        "data": {
            "root_dir": "/workspace/data",
            "input_dst_dir": "/workspace/data/input/dst",
            "label_dst_dir": "/workspace/data/label/dst",
            "predictions_dir": "/workspace/predictions",
            "figures_dir": "/workspace/results/figures",
        },
        "experiments": exp_dict,
        "kbrs": {"weights": {"density": 0.3, "centeredness": 0.3, "mixture": 3.0}},
        "animation": {"fps": 24, "prefetch": 8, "use_processes": True},
    }
    
    
def load_ground_truth(label_dir, target_replays, label_method):
    """
    target_replays 리스트에 있는 리플레이 폴더만 선택하여 GT를 로드합니다.
    """
    candidate_label_dirs = [
        label_dir,
        "/mnt/nas/baecm/starcraft-vision/data/label/dst",
        "/workspace/data/label/dst",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "label", "dst")),
    ]
    actual_label_dir = None
    for d in candidate_label_dirs:
        if d and os.path.exists(d):
            actual_label_dir = d
            break

    if not actual_label_dir:
        actual_label_dir = label_dir

    all_gt = []
    for rep in target_replays:
        rep_clean = str(rep).replace(".rep", "").strip()
        possible_paths = [
            os.path.join(actual_label_dir, f"{rep_clean}.rep", f"{label_method}.json"),
            os.path.join(actual_label_dir, str(rep), f"{label_method}.json"),
        ]
        json_path = None
        for p in possible_paths:
            if os.path.exists(p):
                json_path = p
                break

        if json_path and os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'annotations' in data:
                    for ann in data['annotations']:
                        ann['replay'] = f"{rep_clean}.rep"
                    all_gt.extend(data['annotations'])
    
    if not all_gt:
        return pd.DataFrame()
        
    df = pd.DataFrame(all_gt)
    return df


def load_experiment_json(predictions_dir, pattern, epoch, label_method, target_replay=None):
    """
    JSON 파일 내의 annotations 리스트를 파싱하여 DataFrame으로 반환합니다.
    target_replay가 제공되면 해당 리플레이 JSON 1개만 고속 직격 로드합니다.
    """
    base_dirs = [predictions_dir]
    if os.path.exists("/mnt/nas/baecm/starcraft-vision/predictions") and "/mnt/nas/baecm/starcraft-vision/predictions" not in base_dirs:
        base_dirs.append("/mnt/nas/baecm/starcraft-vision/predictions")

    clean_pattern = pattern.strip("*")
    matching_dirs = []
    for base_d in base_dirs:
        if not os.path.exists(base_d):
            continue

        # 1. Direct path check (5ms ultra-fast bypass)
        direct_path = os.path.join(base_d, clean_pattern)
        if os.path.isdir(direct_path) and os.path.isdir(os.path.join(direct_path, f"model_{epoch:03d}")):
            matching_dirs.append(direct_path)
        else:
            # 2. Glob fallback only if pattern is a wildcard
            matched = glob.glob(os.path.join(base_d, pattern))
            for m in matched:
                epoch_path = os.path.join(m, f"model_{epoch:03d}")
                if os.path.isdir(epoch_path):
                    matching_dirs.append(m)

    if not matching_dirs:
        return pd.DataFrame()

    matching_dirs = sorted(list(set(matching_dirs)), key=os.path.getmtime, reverse=True)
    target_dir = matching_dirs[0]

    if target_replay is not None:
        if isinstance(target_replay, (list, tuple, set)):
            target_reps = [str(r).replace(".rep", "").strip() + ".rep" for r in target_replay]
        else:
            target_reps = [str(target_replay).replace(".rep", "").strip() + ".rep"]
        
        json_files = []
        for r_name in target_reps:
            jp = os.path.join(target_dir, f"model_{epoch:03d}", r_name, f"{label_method}.json")
            if os.path.exists(jp):
                json_files.append(jp)
    else:
        json_files = glob.glob(os.path.join(target_dir, f"model_{epoch:03d}", "*.rep", f"{label_method}.json"))

    if not json_files:
        return pd.DataFrame()

    all_annotations = []
    for f in json_files:
        replay_name = os.path.basename(os.path.dirname(f))
        with open(f, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
            if 'annotations' in data:
                for ann in data['annotations']:
                    ann['replay'] = replay_name
                all_annotations.extend(data['annotations'])

    return pd.DataFrame(all_annotations) if all_annotations else pd.DataFrame()


def prepare_merged_dataframe(gt_df, vanilla_df, kbrs_df):
    """
    Ground Truth, Vanilla, KBRS 데이터프레임을 인덱스(['replay', 'image_id']) 기준으로 병합하고,
    빈 프레임 구간을 bfill()로 메웁니다.
    """
    # 불필요한 접두사 제거 및 멀티 인덱스 설정
    gt = (gt_df.set_index(['replay', 'image_id'])
               .rename(columns=lambda c: c.replace('gt_', '')))
    
    van = (vanilla_df.set_index(['replay', 'image_id'])
                     .rename(columns=lambda c: c.replace('pred_', '')))
    
    kbrs = (kbrs_df.set_index(['replay', 'image_id'])
                   .rename(columns=lambda c: c.replace('pred_', '')))
    
    # 축으로 합치며 상위 레벨 키 부여
    merged = pd.concat({'gt': gt, 'vanilla': van, 'kbrs': kbrs}, axis=1)
    merged = merged.sort_index(axis=1)
    
    # 정렬 보장
    merged = merged.sort_index(level=["replay", "image_id"])
    
    # kbrs / vanilla 블록만 채우기 (gt는 그대로)
    idx = pd.IndexSlice
    cols_to_bfill = merged.loc[:, idx[["kbrs", "vanilla"], :]].columns
    
    merged.loc[:, cols_to_bfill] = (
        merged.loc[:, cols_to_bfill]
          .groupby(level="replay", as_index=False)   # 같은 replay 안에서만
          .bfill()                                   # 다음 프레임의 값으로 메움
    )
    
    merged = merged.dropna().reset_index()
    return merged

def collect_keys_by_level(obj, level=0, result=None):
    """
    중첩된 dict (그리고 dict가 들어 있는 list)에서
    level별로 key를 수집해서 {level: set(keys)} 형태로 반환합니다.
    """
    if result is None:
        result = {}

    if isinstance(obj, dict):
        result.setdefault(level, set())
        for k, v in obj.items():
            result[level].add(k)
            if isinstance(v, (dict, list)):
                collect_keys_by_level(v, level + 1, result)

    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                collect_keys_by_level(item, level, result)

    return result

def load_coco_json(json_path):
    """
    COCO 포맷의 Annotation JSON 파일을 로드합니다.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def calculate_box_iou(box1, box2):
    """
    [x, y, w, h] 포맷의 두 박스 간의 일반적인 IoU를 계산합니다.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    ix1 = max(x1, x2)
    iy1 = max(y1, y2)
    ix2 = min(x1 + w1, x2 + w2)
    iy2 = min(y1 + h1, y2 + h2)
    
    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter_area = inter_w * inter_h
    
    union_area = (w1 * h1) + (w2 * h2) - inter_area
    return inter_area / union_area if union_area > 0 else 0

def get_multi_region_metrics_df(gt_df, pred_df, top_k=3, iou_threshold=0.3):
    """
    상위 K개의 예측(Top-K)을 사용하여 다중 지역 탐지 성능(Recall, Precision)을 계산합니다.
    """
    # 1. Prediction을 Top-K 개수만큼 추출
    pred_topk = pred_df.sort_values(by=['replay', 'image_id', 'score'], ascending=[True, True, False])
    pred_topk = pred_topk.groupby(['replay', 'image_id']).head(top_k)
    
    # 2. 그룹화
    gt_grouped = gt_df.groupby(['replay', 'image_id'])['bbox'].apply(list).to_dict()
    pred_grouped = pred_topk.groupby(['replay', 'image_id'])['bbox'].apply(list).to_dict()
    
    results = []
    
    for (rep, iid), gts in gt_grouped.items():
        preds = pred_grouped.get((rep, iid), [])
        
        if not preds or not gts:
            results.append({
                'replay': rep, 'image_id': iid,
                f'recall@{top_k}': 0.0, f'precision@{top_k}': 0.0
            })
            continue
            
        # Recall: 각 GT마다 적중한 예측 박스가 하나라도 있는지 확인
        gt_hits = 0
        for gt in gts:
            if any(calculate_box_iou(gt, p) >= iou_threshold for p in preds):
                gt_hits += 1
        recall = gt_hits / len(gts)
        
        # Precision: 각 예측 박스가 적중한 GT가 하나라도 있는지 확인
        pred_hits = 0
        for p in preds:
            if any(calculate_box_iou(gt, p) >= iou_threshold for gt in gts):
                pred_hits += 1
        precision = pred_hits / len(preds)
        
        results.append({
            'replay': rep,
            'image_id': iid,
            f'recall@{top_k}': recall,
            f'precision@{top_k}': precision
        })
        
    return pd.DataFrame(results)