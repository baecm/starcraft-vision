import numpy as np
import torch
import torch.nn.functional as F
from collections import namedtuple
from typing import Callable, List, Dict

# 결과 region을 표현하는 구조체
Region = namedtuple("Region", ["bbox", "score", "meta"])


# 개별 score function 정의
def score_density(patch: np.ndarray) -> float:
    """전체 값 합산"""
    return patch.sum()

def score_mixture(patch: np.ndarray) -> float:
    """활성 채널 수 (하나라도 값이 있는 채널 개수)"""
    channel_active = (patch.sum(axis=(1, 2)) > 0).astype(np.uint8)
    return channel_active.sum()

def score_centeredness(patch: np.ndarray) -> float:
    """중심 집중도 (중심 근처에 값이 많을수록 높음)"""
    c, h, w = patch.shape
    y, x = np.mgrid[0:h, 0:w]
    cy, cx = h // 2, w // 2
    sigma = h / 4
    weight = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))  # (h, w)
    return (patch * weight[None, :, :]).sum()

# 종합 score function 생성기
def build_composite_score_fn(
    score_funcs: Dict[str, Callable[[np.ndarray], float]],
    weights: Dict[str, float]
) -> Callable[[np.ndarray], float]:
    """
    가변 score function과 weight dictionary를 받아 종합 score function 생성
    """
    def fn(patch: np.ndarray) -> float:
        total = 0.0
        for name, func in score_funcs.items():
            weight = weights.get(name, 0.0)
            total += weight * func(patch)
        return total
    return fn

# k-BRS region 추출 함수
def extract_kbrs_regions_from_tensor(
    data: np.ndarray,  # shape: (C, H, W)
    region_size: tuple,
    top_k: int = 5,
    stride: int = 1,
    score_fn: Callable[[np.ndarray], float] = None
) -> List[Region]:
    """
    텐서에서 Top-k region을 추출하는 k-BRS 알고리즘

    Args:
        data: 입력 텐서 (채널, 높이, 너비)
        region_size: 추출할 윈도우 크기 (h, w)
        top_k: 추출할 region 개수
        stride: 슬라이딩 간격
        score_fn: patch (C, h, w)에 대한 스코어 함수

    Returns:
        List[Region]
    """
    C, H, W = data.shape
    h, w = region_size

    if score_fn is None:
        score_fn = lambda patch: patch.sum()

    regions = []
    for y in range(0, H - h + 1, stride):
        for x in range(0, W - w + 1, stride):
            patch = data[:, y:y + h, x:x + w]
            score = score_fn(patch)

            regions.append(Region(
                bbox=[x, y, x + w, y + h],
                score=score,
                meta={
                    "patch_sum": patch.sum(),
                    "center": (x + w // 2, y + h // 2),
                    "coords": (x, y),
                }
            ))

    regions.sort(key=lambda r: r.score, reverse=True)
    return regions[:top_k]


def extract_kbrs_regions_fast(
    data: np.ndarray,  # shape: (C, H, W)
    region_size: tuple,
    top_k: int = 5,
    score_weights: Dict[str, float] = None,
    device: str = "cpu"
) -> List[Region]:
    """
    텐서에서 Top-k region을 추출하는 k-BRS 알고리즘 (컨볼루션 기반 최적화)

    Args:
        data: 입력 텐서 (채널, 높이, 너비)
        region_size: 추출할 윈도우 크기 (h, w)
        top_k: 추출할 region 개수
        score_weights: 각 점수 함수의 가중치
        device: 연산을 수행할 장치 ('cpu' 또는 'cuda')

    Returns:
        List[Region]
    """
    h, w = region_size
    C, H, W = data.shape
    
    # 데이터를 PyTorch 텐서로 변환하고 지정된 장치로 이동
    base_tensor = torch.from_numpy(data).unsqueeze(0).float().to(device)  # (1, C, H, W)

    total_score_map = torch.zeros((1, H - h + 1, W - w + 1), device=device)

    # 1. Density Score Map 계산
    if score_weights.get("density", 0.0) > 0:
        # 모든 채널을 합산한 후 컨볼루션
        summed_tensor = base_tensor.sum(dim=1, keepdim=True)  # (1, 1, H, W)
        kernel = torch.ones((1, 1, h, w), device=device)
        density_map = F.conv2d(summed_tensor, kernel, stride=1)
        total_score_map += score_weights["density"] * density_map.squeeze(0)

    # 2. Mixture Score Map 계산
    if score_weights.get("mixture", 0.0) > 0:
        # 각 채널별로 컨볼루션을 적용하여 활성 여부 판단
        kernel = torch.ones((C, 1, h, w), device=device)
        channel_sums = F.conv2d(base_tensor, kernel, stride=1, groups=C) # (1, C, H-h+1, W-w+1)
        mixture_map = (channel_sums > 0).float().sum(dim=1) # (1, H-h+1, W-w+1)
        total_score_map += score_weights["mixture"] * mixture_map

    # 3. Centeredness Score Map 계산
    if score_weights.get("centeredness", 0.0) > 0:
        y, x = torch.mgrid[0:h, 0:w]
        cy, cx = h // 2, w // 2
        sigma = h / 4
        weight = torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2)).to(device) # (h, w)
        
        # 가중치 커널로 컨볼루션
        kernel = weight.unsqueeze(0).unsqueeze(0) # (1, 1, h, w)
        summed_tensor = base_tensor.sum(dim=1, keepdim=True) # (1, 1, H, W)
        centeredness_map = F.conv2d(summed_tensor, kernel, stride=1)
        total_score_map += score_weights["centeredness"] * centeredness_map.squeeze(0)

    # Top-k 점수 및 위치 찾기
    flat_scores = total_score_map.flatten()
    scores, indices = torch.topk(flat_scores, k=top_k, largest=True)
    
    # 2D 인덱스로 변환
    score_map_w = W - w + 1
    coords_y = indices // score_map_w
    coords_x = indices % score_map_w

    # 결과 Region 객체 생성
    regions = []
    for i in range(top_k):
        x, y = coords_x[i].item(), coords_y[i].item()
        score = scores[i].item()
        
        regions.append(Region(
            bbox=[x, y, x + w, y + h],
            score=score,
            meta={
                "center": (x + w // 2, y + h // 2),
                "coords": (x, y),
            }
        ))
        
    return regions

# 사용 예시
if __name__ == "__main__":
    # 예시 입력 텐서: 9채널, 128x128
    np.random.seed(0)
    tensor = np.random.randint(0, 2, size=(9, 128, 128))  # 이진 마스크

    # Score 구성 요소와 가중치
    score_funcs = {
        "density": score_density,
        "mixture": score_mixture,
        "centeredness": score_centeredness
    }
    weights = {
        "density": 1.0,
        "mixture": 0.7,
        "centeredness": 1.2
    }

    score_fn = build_composite_score_fn(score_funcs, weights)

    # Region 추출
    top_regions = extract_kbrs_regions_from_tensor(
        tensor,
        region_size=(20, 12),
        top_k=5,
        stride=1,
        score_fn=score_fn
    )

    for r in top_regions:
        print(f"[Region] BBox: {r.bbox}, Score: {r.score:.2f}, Center: {r.meta['center']}")

    # --- 최적화된 함수 테스트 ---
    print("\n--- Testing Optimized Function (stride=1) ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    import time
    start_time = time.time()
    
    top_regions_fast = extract_kbrs_regions_fast(
        tensor,
        region_size=(20, 12),
        top_k=5,
        score_weights=weights,
        device=device
    )

    end_time = time.time()

    for r in top_regions_fast:
        print(f"[Fast Region] BBox: {r.bbox}, Score: {r.score:.2f}, Center: {r.meta['center']}")
    
    print(f"Optimized function took: {end_time - start_time:.4f} seconds")
