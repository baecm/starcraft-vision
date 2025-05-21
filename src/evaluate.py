import os
import csv
import argparse
import glob
import numpy as np
import pandas as pd
import config

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
    default_data = os.path.join(project_root, 'data')
    default_models = os.path.join(project_root, 'models')

    parser = argparse.ArgumentParser(
        description="Evaluate predicted viewports against human annotations."
    )
    parser.add_argument(
        '--set', type=str, required=True, choices=SET_REPLAYS.keys(),
        help="Select which set to evaluate (set_0, set_1, set_2)"
    )
    parser.add_argument(
        '--label-method', type=str, required=True, choices=config.LABEL_METHODS,
        help="Label extraction method (subfolder inside each .rep directory)"
    )
    parser.add_argument(
        '--window-size', type=int, default=1,
        help="Window size used during training (for model folder prefix)"
    )
    parser.add_argument(
        '--batch-size', type=int, default=32,
        help="Batch size used during training (for model folder prefix)"
    )
    parser.add_argument(
        '--model-number', type=int, default=0,
        help="Checkpoint number to evaluate (logging only)"
    )
    parser.add_argument(
        '--data-root', type=str, default=default_data,
        help="Root directory for data (must contain 'label' subdir)"
    )
    parser.add_argument(
        '--model-root', type=str, default=default_models,
        help="Root directory for model checkpoints"
    )
    parser.add_argument(
        '--partial-length', type=float, default=1.0,
        help="Fraction of each replay to evaluate (0.0 - 1.0)"
    )
    parser.add_argument(
        '--out-csv', type=str, default='./temp.csv',
        help="Path to output CSV file"
    )
    return parser.parse_args()


def find_model_folder(model_root, label_method, window_size, batch_size):
    # Model folders are named like: <label_method>_win<window_size>_b<batch_size>_YYYYMMDD_HHMMSS
    prefix = f"{label_method}_win{window_size}_b{batch_size}"
    pattern = os.path.join(model_root, f"{prefix}*")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No model folder matching '{prefix}*' under {model_root}")
    return os.path.basename(matches[0])


def load_viewport_data(test_names, annotator_dir, label_method):
    """
    Load and concatenate all .vpds.npy files per replay:
      annotator_dir/<replay>.rep/<label_method>/*.vpds.npy
    Returns list of DataFrames and frame counts.
    """
    data_list = []
    frame_list = []

    for replay in test_names:
        rep_folder = os.path.join(annotator_dir, f"{replay}.rep", label_method)
        if not os.path.isdir(rep_folder):
            raise FileNotFoundError(f"Missing directory: {rep_folder}")
        npy_files = sorted(glob.glob(os.path.join(rep_folder, '*.vpds.npy')))
        if not npy_files:
            raise FileNotFoundError(f"No '.vpds.npy' files found in {rep_folder}")

        arrs = []
        for npy_file in npy_files:
            arr = np.load(npy_file)
            # Fix orientation
            if arr.ndim == 2 and arr.shape[1] != 2 and arr.shape[0] == 2:
                arr = arr.T
            if arr.ndim == 1:
                arr = arr.reshape(-1, 2)
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError(f"Unexpected array shape {arr.shape} in {npy_file}")
            arrs.append(arr)

        full_arr = np.vstack(arrs)
        df = pd.DataFrame(full_arr, columns=['vpx', 'vpy'])
        data_list.append(df)
        frame_list.append(len(df))

    return data_list, frame_list


def evaluate_intersection(test_names, annotations, lengths):
    x_len, y_len = 20, 12
    width, height = 128, 128
    max_x, max_y = 3456, 3720

    results = {'i_any': [], 'i_30': [], 'i_50': []}
    for idx, replay in enumerate(test_names):
        dfs = [ann[idx] for ann in annotations]
        limit = lengths[idx]
        overlaps = []
        for t in range(limit):
            canvas = np.zeros((width, height), dtype=int)
            # accumulate other annotators
            for df in dfs[1:]:
                x = int(df.loc[t, 'vpx'] / max_x * (width - x_len))
                y = int(df.loc[t, 'vpy'] / max_y * (height - y_len))
                canvas[x:x+x_len, y:y+y_len] += 1
            # evaluate at index 0
            rx = int(dfs[0].loc[t, 'vpx'] / max_x * (width - x_len))
            ry = int(dfs[0].loc[t, 'vpy'] / max_y * (height - y_len))
            patch = canvas[rx:rx+x_len, ry:ry+y_len]
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
        group = [model_folder] + [h for h in humans if h != human]
        annotations = []
        lengths = [float('inf')] * len(tests)
        for name in group:
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
