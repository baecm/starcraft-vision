import numpy as np
from collections import namedtuple
from typing import Callable, List

# 결과 region을 표현하는 구조체
Region = namedtuple("Region", ["bbox", "score", "meta"])


def build_composite_score_fn(alpha=1.0, beta=0.5, gamma=1.0) -> Callable[[np.ndarray], float]:
    """
    (채널, 높이, 너비) patch에 대해 밀도, 혼합도, 중심 집중도를 조합한 점수 함수 생성기
    """
    def fn(patch: np.ndarray) -> float:
        A, h, w = patch.shape

        # 밀도: 단순합
        density = patch.sum()

        # 혼합도: 각 채널이 하나라도 값을 가진 경우의 개수
        channel_active = (patch.sum(axis=(1, 2)) > 0).astype(np.uint8)
        mixture = channel_active.sum()

        # 중심 집중도: 중심부에 가까울수록 가중치 부여
        y, x = np.mgrid[0:h, 0:w]
        cy, cx = h // 2, w // 2
        sigma = h / 4
        weight = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))  # (h, w)
        centered = (patch * weight[None, :, :]).sum()

        return alpha * density + beta * mixture + gamma * centered

    return fn


def extract_kbrs_regions_from_tensor(
    data: np.ndarray,  # shape: (A, H, W)
    region_size: tuple,
    top_k: int = 5,
    stride: int = 16,
    score_fn: Callable[[np.ndarray], float] = None
) -> List[Region]:
    """
    (A, H, W) 텐서에서 Top-k region을 추출하는 k-BRS 알고리즘

    Args:
        data: 입력 텐서 (채널, 높이, 너비)
        region_size: 추출할 윈도우 크기 (h, w)
        top_k: 추출할 region 개수
        stride: 슬라이딩 간격
        score_fn: patch (A, h, w)에 대한 스코어 함수

    Returns:
        List[Region]
    """
    A, H, W = data.shape
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
    # 예시 입력 텐서: 9채널, 480x640
    np.random.seed(0)
    tensor = np.random.randint(0, 2, size=(9, 480, 640))  # 이진 마스크

    score_fn = build_composite_score_fn(alpha=1.0, beta=0.7, gamma=1.2)

    top_regions = extract_kbrs_regions_from_tensor(
        tensor,
        region_size=(64, 64),
        top_k=5,
        stride=16,
        score_fn=score_fn
    )

    for r in top_regions:
        print(f"[Region] BBox: {r.bbox}, Score: {r.score:.2f}, Center: {r.meta['center']}")
