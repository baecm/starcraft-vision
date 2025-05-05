import os
import csv
import argparse
import numpy as np
import pandas as pd

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
    parser = argparse.ArgumentParser(description="Evaluate predicted viewports against human annotations.")
    parser.add_argument('--set', type=str, required=True, choices=SET_REPLAYS.keys(), help="Set to evaluate (set_0/set_1/set_2)")
    parser.add_argument('--model-name', type=str, required=True, help="Model name to evaluate")
    parser.add_argument('--model-number', type=int, default=0, help="Model number to evaluate")
    parser.add_argument('--model-path', type=str, default='./models/', help="Path to the model directory")
    parser.add_argument('--partial-length', type=float, default=1.0, help="Fraction of each replay to evaluate (0~1)")
    parser.add_argument('--out-csv', type=str, default='./temp.csv', help="Path to output CSV for saving evaluation results")
    return parser.parse_args()


def load_viewport_data(test_names, label_path):
    data_arr, frame_arr = [], []
    for name in test_names:
        path = os.path.join(label_path, f"{name}.rep.vpd")
        df = pd.read_csv(path)
        terminal_frame = int(df['frame'].iloc[-1])
        df = df.reindex(range(terminal_frame)).fillna(method='ffill').reset_index()
        df['vpx'] = df['vpx'].astype(int)
        df['vpy'] = df['vpy'].astype(int)
        data_arr.append(df)
        frame_arr.append(terminal_frame)
    return data_arr, frame_arr


def evaluate_intersection(test_names, labels_arr, min_length_arr):
    x_len, y_len = 20, 12
    width, height = 128, 128
    max_x, max_y = 3456, 3720
    total_intersection, multi_intersection = [], []
    is_intersect, is_intersect_30, is_intersect_50 = [], [], []

    for j, name in enumerate(test_names):
        label_arr = [item[j] for item in labels_arr]
        min_len = min_length_arr[j]
        temp_result = []

        for i in range(min_len):
            canvas = np.zeros((width, height))
            for labels in label_arr[1:]:
                x = int(round(labels['vpx'][i] / max_x * (width - x_len)))
                y = int(round(labels['vpy'][i] / max_y * (height - y_len)))
                canvas[x:x + x_len, y:y + y_len] += 1

            px = int(round(label_arr[0]['vpx'][i] / max_x * (width - x_len)))
            py = int(round(label_arr[0]['vpy'][i] / max_y * (height - y_len)))
            patch = canvas[px:px + x_len, py:py + y_len]

            multi = np.mean(patch)
            binary = np.mean(np.where(patch > 0, 1, 0))
            multi_intersection.append(multi)
            total_intersection.append(binary)
            temp_result.append(binary)
            is_intersect.append(int(binary > 0))
            is_intersect_30.append(int(binary >= 0.3))
            is_intersect_50.append(int(binary >= 0.5))

        print(f"{name}: {np.mean(temp_result):.4f}")

    return total_intersection, multi_intersection, is_intersect, is_intersect_30, is_intersect_50


def run_evaluate(args):
    test_names = SET_REPLAYS[args.set]
    human_names = SET_USERS[args.set]
    model_name = args.model_name
    partial = args.partial_length
    
    print("=== Evaluation Start ===")
    print(f"Model: {model_name}, Set: {args.set}")
    print(f"Test names: {test_names}")
    print(f"Human names: {human_names}")
    print(f"Model number: {args.model_number}")
    print(f"Output CSV: {args.out_csv}")
    print(f"Partial length: {partial:.2f}")

    # 평가 루프
    total_i, total_m, total_0, total_30, total_50 = [], [], [], [], []

    for k in human_names:
        label_names = [model_name] + [h for h in human_names if h != k]
        labels_arr = []
        min_length_arr = [np.inf] * len(test_names)

        for name in label_names:
            label_path = f"./labels/{name}/"
            label_data, frame_lengths = load_viewport_data(test_names, label_path)
            labels_arr.append(label_data)
            for j in range(len(test_names)):
                min_length_arr[j] = min(min_length_arr[j], int(frame_lengths[j] * partial))

        result = evaluate_intersection(test_names, labels_arr, min_length_arr)
        i, m, i0, i30, i50 = map(np.mean, result)

        print(f"{k}: i={i:.4f}, m={m:.4f}, 0/30/50 = {np.mean(i0):.2f}, {np.mean(i30):.2f}, {np.mean(i50):.2f}")
        total_i.append(i)
        total_m.append(m)
        total_0.append(np.mean(i0))
        total_30.append(np.mean(i30))
        total_50.append(np.mean(i50))

    print("=== Final Result ===")
    print(f"Intersection: {np.mean(total_i):.4f}, Multi: {np.mean(total_m):.4f}")
    print(f"0/30/50: {np.mean(total_0):.2f}, {np.mean(total_30):.2f}, {np.mean(total_50):.2f}")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, 'a') as f:
        csv.writer(f).writerow([
            f"{np.mean(total_0):.4f}",
            f"{np.mean(total_30):.4f}",
            f"{np.mean(total_50):.4f}"
        ])


def main():
    args = parse_arguments()
    run_evaluate(args)


if __name__ == "__main__":
    main()
