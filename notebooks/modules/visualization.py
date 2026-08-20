import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.cm as cm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from modules.constants import Channel, GRID_W, GRID_H
from modules.kbrs_metrics import score_density, score_centeredness, score_mixture, _worker_compute, _extract_gts

DPI = 300

def overview(x: np.ndarray, title=None):
    """
    내부적으로 figure를 생성하고 오버뷰를 그립니다.
    """
    fig, ax = plt.subplots(figsize=(6, 6), dpi=DPI)
    
    # Terrain & resource
    ax.imshow(x[Channel.Terrain.value] > 0, cmap='Greys', alpha=0.5)
    resource_alpha = np.where(x[Channel.Resource.value] == 1, 1.0, 0.0)
    ax.imshow(x[Channel.Resource.value] == 1, cmap='BuGn', alpha=resource_alpha)
    
    # Player colors
    for ch_p1, ch_p2 in [
        (Channel.Player_1_Worker, Channel.Player_2_Worker),
        (Channel.Player_1_Ground, Channel.Player_2_Ground),
        (Channel.Player_1_Air, Channel.Player_2_Air),
        (Channel.Player_1_Building, Channel.Player_2_Building)
    ]:
        p1_alpha = np.where(x[ch_p1.value] != 0, 1.0, 0.0)
        p2_alpha = np.where(x[ch_p2.value] != 0, 1.0, 0.0)
        ax.imshow(x[ch_p1.value] != 0, cmap='Greens', alpha=p1_alpha)
        ax.imshow(x[ch_p2.value] != 0, cmap='Reds', alpha=p2_alpha)

    # Vision mask
    vision_alpha = np.where(x[Channel.Vision.value] == 1, 0.0, 0.9)
    ax.imshow(np.zeros_like(x[Channel.Vision.value]), cmap='Greys_r', alpha=vision_alpha)
    
    if title: ax.set_title(title)
    # ax.axis('off')
    return fig, ax


def feature(input_dst_dir, replay, image_id, gt=None, vanilla=None, kbrs=None, save_dir=None):
    """
    item 전체를 넘기지 않고, 필요한 값들만 명시적으로 전달합니다.
    gt, vanilla, kbrs가 없으면(None) 해당 마킹은 생략합니다.
    """
    item_npy_path = os.path.join(input_dst_dir, str(replay), f"{image_id}.npy")
    item_npy = np.load(item_npy_path)
    
    fig, axes = plt.subplots(1, 3, dpi=DPI, figsize=(12, 8))    
    
    den = score_density(item_npy)
    cen = score_centeredness(item_npy)
    mix = score_mixture(item_npy)
    score = 0.3 * den + 0.3 * cen + 3.0 * mix

    data_map = {'gt': gt, 'vanilla': vanilla, 'kbrs': kbrs}
    colors = {'gt': 'red', 'vanilla': 'blue', 'kbrs': 'purple'}

    axes[0].imshow(den)
    axes[0].set_title('density')
    
    axes[1].imshow(cen)
    axes[1].set_title('centeredness')
    
    axes[2].imshow(mix)
    axes[2].set_title('mixture')
    
    for ax in axes:
        ax.set_xlim(0, GRID_W)
        ax.set_ylim(GRID_H, 0)

    plt.show()
    return fig, axes


def channel(data, save_dir=None, rep_name=None, frame=None):
    """
    각 채널별 시각화 결과를 리스트 형태로 반환합니다.
    save_dir, rep_name, frame이 주어지면 저장도 병행합니다.
    """
    p1_color = cm.get_cmap('Greens')(0.7)
    p2_color = cm.get_cmap('Reds')(0.7)
    resource_color = cm.get_cmap('GnBu')(0.7)
    
    fig, axes = plt.subplots(2, 6, figsize=(12, 5), dpi=DPI)
    
    axes[0][0].imshow(data[Channel.Player_1_Worker.value] != 0, cmap='Greens')
    axes[0][0].set_title(Channel.Player_1_Worker.name)
    axes[0][1].imshow(data[Channel.Player_1_Ground.value] != 0, cmap='Greens')
    axes[0][1].set_title(Channel.Player_1_Ground.name)
    axes[0][2].imshow(data[Channel.Player_1_Air.value] != 0, cmap='Greens')
    axes[0][2].set_title(Channel.Player_1_Air.name)
    axes[0][3].imshow(data[Channel.Player_1_Building.value] != 0, cmap='Greens')
    axes[0][3].set_title(Channel.Player_1_Building.name)

    axes[1][0].imshow(data[Channel.Player_1_Worker.value] != 0, cmap='Greens')
    axes[1][0].set_title(Channel.Player_1_Worker.name)
    axes[1][1].imshow(data[Channel.Player_2_Ground.value] != 0, cmap='Reds')
    axes[1][1].set_title(Channel.Player_2_Ground.name)
    axes[1][2].imshow(data[Channel.Player_2_Air.value] != 0, cmap='Reds')
    axes[1][2].set_title(Channel.Player_2_Air.name)
    axes[1][3].imshow(data[Channel.Player_2_Building.value] != 0, cmap='Reds')
    axes[1][3].set_title(Channel.Player_2_Building.name)
    
    axes[0][4].imshow(data[Channel.Resource.value] != 0, cmap='BuGn')
    axes[0][4].set_title(Channel.Resource.name)
    axes[0][5].imshow(data[Channel.Terrain.value] != 0, cmap='viridis')
    axes[0][5].set_title(Channel.Terrain.name)
    axes[1][4].imshow(data[Channel.Vision.value] != 0, cmap='gray')
    axes[1][4].set_title(Channel.Vision.name)
    
    axes[1][5].axis('off')  # 마지막 subplot은 비워둡니다.
    
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
        
    return fig, axes


def render_frame_with_viewport_overlay(
    input_npy: np.ndarray,
    gt_bboxes=None,
    pred_bboxes=None,
    title=None,
    ax=None,
    figsize=(8, 8),
    show_legend=True,
):
    """
    11개 입력 채널 npy 데이터로 게임 미니맵 상태(State)를 복원(Reconstruction)하고,
    GT 뷰포트(Red Dashed) 및 모델 예측 뷰포트(Blue/Green/Cyan Solid)를 
    오버레이하여 실제 예측 배치 결과를 시각화합니다.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
    else:
        fig = ax.figure

    # 1. 게임 미니맵 프레임 복원 (Terrain, Resource, P1, P2, Vision)
    ax.imshow(input_npy[Channel.Terrain.value] > 0, cmap='Greys', alpha=0.5)
    res_alpha = np.where(input_npy[Channel.Resource.value] == 1, 1.0, 0.0)
    ax.imshow(input_npy[Channel.Resource.value] == 1, cmap='BuGn', alpha=res_alpha)

    # Player 1 (Greens) & Player 2 (Reds)
    for ch_p1, ch_p2 in [
        (Channel.Player_1_Worker, Channel.Player_2_Worker),
        (Channel.Player_1_Ground, Channel.Player_2_Ground),
        (Channel.Player_1_Air, Channel.Player_2_Air),
        (Channel.Player_1_Building, Channel.Player_2_Building)
    ]:
        p1_a = np.where(input_npy[ch_p1.value] != 0, 1.0, 0.0)
        p2_a = np.where(input_npy[ch_p2.value] != 0, 1.0, 0.0)
        ax.imshow(input_npy[ch_p1.value] != 0, cmap='Greens', alpha=p1_a)
        ax.imshow(input_npy[ch_p2.value] != 0, cmap='Reds', alpha=p2_a)

    vis_alpha = np.where(input_npy[Channel.Vision.value] == 1, 0.0, 0.85)
    ax.imshow(np.zeros_like(input_npy[Channel.Vision.value]), cmap='Greys_r', alpha=vis_alpha)

    # 2. GT 뷰포트 오버레이 (빨간색 점선 박스)
    if gt_bboxes:
        for idx, box in enumerate(gt_bboxes):
            x, y, w, h = box[:4]
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=2.0, edgecolor='red', facecolor='none', linestyle='--',
                label='Ground Truth (Rater)' if idx == 0 and show_legend else None
            )
            ax.add_patch(rect)

    # 3. 모델 예측 뷰포트 오버레이 (Top-1: 파란색, Top-2: 초록색, Top-3: 하늘색)
    colors = ['royalblue', 'limegreen', 'darkcyan', 'gold', 'magenta']
    if pred_bboxes:
        for idx, box in enumerate(pred_bboxes):
            x, y, w, h = box[:4]
            score = box[4] if len(box) > 4 else None
            color = colors[idx % len(colors)]
            lbl_name = f"Pred Top-{idx+1}" + (f" ({score:.2f})" if score is not None else "")
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=2.5, edgecolor=color, facecolor='none', linestyle='-',
                label=lbl_name if show_legend else None
            )
            ax.add_patch(rect)

    ax.set_xlim(0, 128)
    ax.set_ylim(128, 0)
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    if show_legend:
        ax.legend(loc='upper right', framealpha=0.8)

    return fig, ax

