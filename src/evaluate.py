import os
import csv
import argparse
import glob
import numpy as np
import pandas as pd
import config
import json

SET_REPLAYS = {
    "set_0": ["36", "212", "438", "522", "1660"],
    "set_1": ["1559", "1628", "2351", "6219", "11251"],
    "set_2": ["275", "1725", "3613", "4520", "4664"]
}

SET_USERS = {
    "set_0": ["bcm_allframes", "yws_allframes", "cyh_allframes", "pdh_allframes", "jht_allframes"],
    "set_1": ["1_allframes", "2_allframes", "3_allframes", "4_allframes", "5_allframes"],
    "set_2": ["6_allframes", "7_allframes", "8_allframes", "9_allframes", "10_allframes"]
}


def parse_arguments():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_root = os.path.join(project_root, 'data')
    model_root = os.path.join(project_root, 'models')

    parser = argparse.ArgumentParser(description="Evaluate predicted viewports against human annotations.")

    # Data and Model Specification
    group_spec = parser.add_argument_group("Data and Model Specification")
    group_spec.add_argument('--set', type=str, required=True, choices=SET_REPLAYS.keys(), help="Select which replay set to evaluate (e.g., set_0, set_1).")
    group_spec.add_argument('--label-method', type=str, required=True, choices=config.LABEL_METHODS, help="Label extraction method.")
    group_spec.add_argument('--data-root', type=str, default=data_root, help="Root directory for data (must contain 'label' subdir).")
    group_spec.add_argument('--model-root', type=str, default=model_root, help="Root directory for model checkpoints.")
    group_spec.add_argument("--include-components", type=str, nargs='+', default=['worker', 'ground', 'air', 'building', 'vision'], help="List of components to include.")

    # Model Hyperparameters (for finding the folder)
    group_hyper = parser.add_argument_group("Model Hyperparameters")
    group_hyper.add_argument('--window-size', type=int, default=1, help="Window size used during training.")
    group_hyper.add_argument('--batch-size', type=int, default=32, help="Batch size used during training.")
    group_hyper.add_argument('--model-number', type=int, default=0, help="Checkpoint number to evaluate (for logging only).")

    # Evaluation Settings
    group_eval = parser.add_argument_group("Evaluation Settings")
    group_eval.add_argument('--partial-length', type=float, default=1.0, help="Fraction of each replay to evaluate (0.0 - 1.0).")
    group_eval.add_argument('--out-csv', type=str, default='./temp.csv', help="Path to output CSV file for results.")

    return parser.parse_args()


def find_model_folder(model_root, label_method, window_size, batch_size, date_string=None):
    # Model folders are named like: <label_method>_win<window_size>_b<batch_size>_YYYYMMDD_HHMMSS
    prefix = f"{label_method}_win{window_size}_b{batch_size}"
    pattern = os.path.join(model_root, f"{prefix}*")
    print(f">> Looking for model folder with prefix: {prefix}")
    print(f">> Searching for model folder with pattern: {pattern}")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No model folder matching '{prefix}*' under {model_root}")
    return os.path.basename(matches[0])


def load_viewport_data(test_names, annotator_dir, label_method):
    """
    → (수정된 버전) 
    Load and concatenate all (x,y) viewport coordinates from each replay's single JSON:
      annotator_dir/<replay>.rep/<label_method>.json
    Returns: list of DataFrames (각 annotator별) 과 frame counts
    """
    data_list = []
    frame_list = []

    for replay in test_names:
        # 1) 단일 JSON 파일 경로
        json_path = os.path.join(annotator_dir, f"{replay}.rep", f"{label_method}.json")
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"Missing JSON: {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            coco = json.load(f)

        # 2) JSON 내 images, annotations 읽기
        #    images: [{"id": frame_id, "file_name": "...", "width": W, "height": H}, ...]
        #    annotations: [{"id": ann_id, "image_id": frame_id, "bbox":[x,y,w,h], ...}, ...]
        images_info = coco.get("images", [])
        anns       = coco.get("annotations", [])

        # 3) frame_id 순서대로 (x,y)를 모을 리스트
        #    - annotation의 bbox에서 (x, y)만 꺼내서 list에 append
        #    - 한 frame에 여러 annotation이 있으면(=여러 사람이 겹쳐 본다면) → 전부 append (멀티 인스턴스)
        per_frame_coords = {}  # frame_id → [(x,y), (x,y), ...]
        for ann in anns:
            frame_id = int(ann["image_id"])
            x, y, w, h = map(int, ann["bbox"])
            # 기존 .vpds.npy가 (x, y)만 제공했으므로, 동일하게 (x,y)만 저장
            per_frame_coords.setdefault(frame_id, []).append((x, y))

        # 4) frames 개수: images_info에 있는 frame_id 중 최고값 + 1 혹은 images_info 길이
        #    실제로 “프레임 수”는 images_info 개수(=json에 등록된 이미지 개수)와 동일하다고 가정
        frame_count = len(images_info)

        # 5) DataFrame 생성: 각 frame마다 (x,y) 좌표들을 순서대로 저장
        #    - 사람이 여러 명일 수도 있으므로, “한 사람당 한 Series”로 관리하지 않고,
        #      “frame별로 x,y만 모아서 DataFrame 생성”해 둔다. (한 annotator가 아니라, 단일 JSON 안의 모든 annotation)
        arrs = []
        for fid in range(frame_count):
            # 만약 해당 frame_id에 annotation이 1개도 없다면, (0,0) 혹은 NaN 처리 → 일단 (0,0)으로 채움
            coords_list = per_frame_coords.get(fid, [(0, 0)])
            # 한 frame에 여러 annotation이 있을 수 있으므로, 평균 좌표로 대표하거나 첫 개체만 택할 수도 있다.
            # 그러나 원본 .vpds.npy처럼 “한 frame당 한 사람”을 기준으로 삼고 싶다면, coords_list[0]만 사용한다.
            x0, y0 = coords_list[0]
            arrs.append((fid, x0, y0))

        df = pd.DataFrame(arrs, columns=['frame', 'vpx', 'vpy'])
        data_list.append(df)
        frame_list.append(frame_count)

    return data_list, frame_list


def evaluate_intersection(test_names, annotations, lengths):
    """
    원본과 동일: 
    x_len, y_len = 20, 12  ← viewport patch 크기 (kernel_shape)
    width, height = 128,128  ← 화면 해상도 (origin_shape)
    max_x, max_y = 3456,3720  ← 전체 맵 좌표 최대값 (기존 코드 기준)
    """
    x_len, y_len = 20, 12
    width, height = 128, 128
    max_x, max_y = 3456, 3720

    results = {'i_any': [], 'i_30': [], 'i_50': []}
    for idx, replay in enumerate(test_names):
        dfs = [ann[idx] for ann in annotations]   # 여러 annotator DataFrame
        limit = lengths[idx]
        overlaps = []

        for t in range(limit):
            canvas = np.zeros((width, height), dtype=int)
            # 1) 먼저 “비교 annotator”들(인덱스 1~N)을 캔버스에 누적
            for df in dfs[1:]:
                vpx = int(df.loc[t, 'vpx'])
                vpy = int(df.loc[t, 'vpy'])
                # 전체 맵 좌표 → 화면 좌표 비례 계산 (기존에 쓰던 방식 그대로)
                x = int(vpx / max_x * (width - x_len))
                y = int(vpy / max_y * (height - y_len))
                canvas[x:x + x_len, y:y + y_len] += 1

            # 2) 기준 annotator(인덱스 0) 위치
            rx = int(dfs[0].loc[t, 'vpx'] / max_x * (width - x_len))
            ry = int(dfs[0].loc[t, 'vpy'] / max_y * (height - y_len))
            patch = canvas[rx:rx + x_len, ry:ry + y_len]
            any_overlap = (patch > 0).mean()

            results['i_any'].append(int(any_overlap > 0))
            results['i_30'].append(int(any_overlap >= 0.3))
            results['i_50'].append(int(any_overlap >= 0.5))
            overlaps.append(any_overlap)

        print(f"{replay}: {np.mean(overlaps):.4f}")

    return results


def run_evaluate(args):
    model_folder = find_model_folder(
        args.model_root,
        args.label_method,
        args.window_size,
        args.batch_size
    )
    print("=== Evaluation Start ===")
    print(
        f"Model folder: {model_folder}, Method: {args.label_method}, "
        f"Checkpoint: {args.model_number}, Set: {args.set}"
    )

    tests = SET_REPLAYS[args.set]
    humans = SET_USERS[args.set]
    partial = args.partial_length

    overall = {'i_any': [], 'i_30': [], 'i_50': []}
    for human in humans:
        # (model_folder는 실제로 사람 annotator 이름이 아니므로, 이 부분은 기존 방식 그대로 두거나
        #  필요한 경우 “사람 annotator 디렉토리”를 가리키도록 수정해야 할 수도 있습니다.)
        group = [model_folder] + [h for h in humans if h != human]
        annotations = []
        lengths = [float('inf')] * len(tests)

        for name in group:
            # annotator_dir가 “data/label/dst” 밑에 있는 replay 폴더 구조를 가리켜야 함
            annotator_dir = os.path.join(args.data_root, 'label', 'dst')
            data, frames = load_viewport_data(
                tests,
                annotator_dir,
                args.label_method
            )
            annotations.append(data)
            lengths = [min(l, int(f * partial)) for l, f in zip(lengths, frames)]

        res = evaluate_intersection(tests, annotations, lengths)
        i_any_avg = np.mean(res['i_any'])
        i_30_avg = np.mean(res['i_30'])
        i_50_avg = np.mean(res['i_50'])
        print(
            f"{human}: i={i_any_avg:.4f}, "
            f"0/30/50 = {i_any_avg:.2f}/{i_30_avg:.2f}/{i_50_avg:.2f}"
        )
        overall['i_any'].append(i_any_avg)
        overall['i_30'].append(i_30_avg)
        overall['i_50'].append(i_50_avg)

    print("=== Final Result ===")
    print(
        f"0/30/50: {np.mean(overall['i_any']):.2f}/"
        f"{np.mean(overall['i_30']):.2f}/"
        f"{np.mean(overall['i_50']):.2f}"
    )

    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    with open(args.out_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            f"{np.mean(overall['i_any']):.4f}",
            f"{np.mean(overall['i_30']):.4f}",
            f"{np.mean(overall['i_50']):.4f}"
        ])


def main():
    args = parse_arguments()
    run_evaluate(args)


if __name__ == '__main__':
    main()