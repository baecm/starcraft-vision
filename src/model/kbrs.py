import numpy as np
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
    stride: int = 16,
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
        region_size=(64, 64),
        top_k=5,
        stride=16,
        score_fn=score_fn
    )

    for r in top_regions:
        print(f"[Region] BBox: {r.bbox}, Score: {r.score:.2f}, Center: {r.meta['center']}")
