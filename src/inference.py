import os
import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from dataset.custom_penn_fudan import CustomPennFudanDataset
from model.maskrcnn_builder import get_model_instance_segmentation
import config

def parse_arguments():
    parser = argparse.ArgumentParser(description="Inference script: run Mask R-CNN on replay data and save vpx/vpy arrays")
    parser.add_argument('--replays', nargs='+', required=True, help="List of replay names (directories under data_root/pair)")
    
    parser.add_argument('--label-method', type=str, default=config.LABEL_METHODS[0], choices=config.LABEL_METHODS, help="Label extraction method")
    parser.add_argument('--window-size', type=int, default=1, help="Window size used during training")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size used during training (for model folder prefix)")
    parser.add_argument('--checkpoint', type=int, default=30, help="Epoch number of the model checkpoint to use")
    
    parser.add_argument('--data-root', type=str, default=os.path.join(os.getcwd(), 'data'), help="Root directory for data (must contain 'pair' and 'label' subdirs)")
    parser.add_argument('--model-root', type=str, default=os.path.join(os.getcwd(), 'models'), help="Root directory for model checkpoints")
    parser.add_argument('--cuda', action='store_true', help="Use CUDA for inference")
    
    return parser.parse_args()


def find_model_folder(model_root, label_method, window_size, batch_size):
    prefix = f"{label_method}_win{window_size}_b{batch_size}"
    candidates = sorted(glob.glob(os.path.join(model_root, f"{prefix}*")))
    if not candidates:
        raise FileNotFoundError(f"No model folder found matching {prefix}* in {model_root}")
    return os.path.basename(candidates[0])


def run_inference(args):
    # locate model folder and checkpoint
    model_folder = find_model_folder(
        args.model_root, args.label_method, args.window_size, args.batch_size
    )
    model_dir = os.path.join(args.model_root, model_folder)
    ckpt_path = os.path.join(model_dir, f"model_{args.checkpoint}.pth")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # prepare device and model
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    num_classes = 2
    model = get_model_instance_segmentation(num_classes, window_size=args.window_size)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    data_pair = os.path.join(args.data_root, 'pair')
    data_label = os.path.join(args.data_root, 'label')

    for replay in args.replays:
        # load .npy frames
        replay_folder = os.path.join(data_pair, f"{replay}.rep", args.label_method)
        npy_files = sorted(glob.glob(os.path.join(replay_folder, '*.npy')))
        if not npy_files:
            raise FileNotFoundError(f"No .npy files found under {replay_folder}")

        vpx_list = []
        vpy_list = []
        step = 8  # frame interval in CSV

        for idx in tqdm(range(len(npy_files)), desc=f"Inferring {replay}"):
            # collect window
            start = max(0, idx - args.window_size + 1)
            files = npy_files[start:idx+1]
            frames = []
            for f in files:
                arr = np.load(f, allow_pickle=True)
                frames.append(arr[0])
            # pad
            while len(frames) < args.window_size:
                frames.insert(0, frames[0])
            stack = np.stack(frames, axis=0)  # shape (T, C, H, W)
            Tt, C, H, W = stack.shape
            img = torch.from_numpy(stack.reshape(Tt * C, H, W)).unsqueeze(0).to(device).float()

            with torch.no_grad():
                pred = model(img)[0]
            boxes = pred['boxes']
            if boxes.shape[0] == 0:
                if idx > 0:
                    vpx, vpy = vpx_list[-1], vpy_list[-1]
                else:
                    vpx, vpy = 0, 0
            else:
                x1, y1 = boxes[0][:2]
                vpx = int(x1.item()) * 32
                vpy = int(y1.item()) * 32
            vpx_list.append(vpx)
            vpy_list.append(vpy)

        # prepare output dir
        out_dir = os.path.join(data_label, model_folder, f"{replay}.rep", args.label_method)
        os.makedirs(out_dir, exist_ok=True)

        # save numpy
        arr = np.stack([vpx_list, vpy_list], axis=1)
        npy_path = os.path.join(out_dir, 'pred.vpds.npy')
        np.save(npy_path, arr)
        print(f"Saved viewport npy: {npy_path}")

        # create .rep.vpd CSV
        import pandas as pd
        frames_idx = np.arange(0, len(vpx_list) * step, step)
        df_pred = pd.DataFrame({
            'frame': frames_idx,
            'vpx': vpx_list,
            'vpy': vpy_list
        })
        all_frames = np.arange(0, frames_idx[-1] + 1)
        df_all = pd.DataFrame({'frame': all_frames})
        df_merged = df_all.merge(df_pred, on='frame', how='left').fillna(method='ffill')
        csv_path = os.path.join(out_dir, f"{replay}.rep.vpd")
        df_merged.to_csv(csv_path, index=False)
        print(f"Saved viewport CSV: {csv_path}")


def main():
    args = parse_arguments()
    run_inference(args)


if __name__ == '__main__':
    main()
