# /workspace/notebooks/utils.py
import os, json, mmap, gzip
from pathlib import Path
from typing import Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from functools import partial

import pandas as pd


# ---------- 경로 수집 (빠른 scandir 재귀) ----------
def _iter_json_paths(root: str) -> List[str]:
    out = []
    stack = [os.path.abspath(root)]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    if e.is_dir(follow_symlinks=False):
                        stack.append(e.path)
                    elif e.is_file() and e.name.endswith(".json"):
                        out.append(e.path)
        except PermissionError:
            pass
    return out


def _build_json_paths(
    directory: str,
    label_method: str = "all_correct",
    replays: Optional[List[str]] = None,
) -> List[str]:
    directory = os.path.abspath(directory)
    if replays:
        paths = [
            os.path.join(directory, rep, f"{label_method}.json") for rep in replays
        ]
        # 존재하는 파일만
        return [p for p in paths if os.path.isfile(p)]
    # 전체 스캔
    return _iter_json_paths(directory)


# ---------- 빠른 로더 (orjson + mmap, fallback 표준 json) ----------
import os, json, mmap, gzip


def _read_one_json(path: str, use_orjson: bool = True, use_mmap: bool = True):
    """
    use_mmap=True: zero-copy 파싱(스레드 경로에서만 사용 권장)
    use_mmap=False: 안전한 bytes 파싱(프로세스 경로에서 강제)
    """
    try:
        size = os.path.getsize(path)
        if size == 0:
            return path, None, "EmptyFileError: file size is 0"

        if use_orjson:
            try:
                import orjson

                with open(path, "rb") as f:
                    # gzip 여부 감지
                    head = f.read(2)
                    f.seek(0)
                    if head == b"\x1f\x8b":
                        with gzip.GzipFile(fileobj=f) as gz:
                            b = gz.read()
                            return path, orjson.loads(b), None

                    if use_mmap:
                        # 스레드 경로: mmap + memoryview
                        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                            # 일부 환경에서 memoryview(mm)와 orjson 조합이 불안정한 사례가 있어
                            # 확실히 하려면 mm[:] 로 bytes 복사 사용(속도 약간↓, 안전성↑)
                            return path, orjson.loads(mm[:]), None
                    else:
                        # 프로세스 경로: 안전하게 bytes로 읽기
                        b = f.read()
                        return path, orjson.loads(b), None
            except ModuleNotFoundError:
                pass  # orjson 없으면 표준 json으로 폴백

        # 표준 json 폴백 (BOM 처리 위해 utf-8-sig)
        with open(path, "r", encoding="utf-8-sig") as f:
            return path, json.load(f), None

    except Exception as e:
        return path, None, f"{type(e).__name__}: {e}"


def _read_one_json_proc(path: str):
    # 프로세스에서는 mmap 비활성화
    return _read_one_json(path, use_orjson=True, use_mmap=False)


# ---------- 자동 전략 선택 ----------
def _choose_strategy(paths: List[str], bytes_threshold: int = 512_000) -> str:
    """
    파일 크기 중앙값이 threshold 이상이면 'process', 아니면 'thread'
    - 작은 파일: 스레드가 유리(I/O 병목 + orjson이 GIL 거의 점유 안함)
    - 큰 파일: 프로세스가 유리(파싱 CPU 부담 분산)
    """
    if not paths:
        return "thread"
    sizes = []
    step = max(1, len(paths) // 200)  # 샘플링
    for p in paths[::step]:
        try:
            sizes.append(os.path.getsize(p))
        except OSError:
            pass
    if not sizes:
        return "thread"
    from statistics import median

    return "process" if median(sizes) >= bytes_threshold else "thread"


# ---------- 공통 실행기 ----------
def read_json_files_fast(
    directory: str,
    label_method: str = "all_correct",
    replays: Optional[List[str]] = None,
    max_workers: Optional[int] = None,
    force: Optional[str] = None,  # 'thread'|'process'|None(auto)
) -> List[Any]:
    paths = _build_json_paths(directory, label_method, replays)
    if not paths:
        return []

    # 전략/워커 수 결정
    strategy = force or _choose_strategy(paths)
    if max_workers is None:
        cpu = os.cpu_count() or 4
        if strategy == "process":
            max_workers = min(cpu, 8)
        else:
            max_workers = min(32, cpu * 2)

    results: List[Any] = []
    errors: List[str] = []

    if strategy == "process":
        chunksize = max(1, len(paths) // (max_workers * 8))
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                # 프로세스에서는 mmap 비활성 래퍼 사용
                for path, data, err in ex.map(
                    _read_one_json_proc, paths, chunksize=chunksize
                ):
                    if err:
                        errors.append(f"[FAIL] {path} -> {err}")
                    else:
                        results.append(data)
        except BrokenProcessPool as e:
            print(f"[WARN] Process pool failed ({e}). Falling back to thread pool.")
            # 스레드로 폴백 (orjson + mmap 사용 가능)
            with ThreadPoolExecutor(
                max_workers=min(32, (os.cpu_count() or 4) * 2),
                thread_name_prefix="json",
            ) as ex:
                for path, data, err in ex.map(_read_one_json, paths):
                    if err:
                        errors.append(f"[FAIL] {path} -> {err}")
                    else:
                        results.append(data)
    else:
        # ThreadPool: orjson(C) 구현이 GIL을 잘 놓아서 보통 매우 빠름
        with ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="json"
        ) as ex:
            for path, data, err in ex.map(_read_one_json, paths):
                if err:
                    errors.append(f"[FAIL] {path} -> {err}")
                else:
                    results.append(data)

    if errors:
        print("\n".join(errors))
    return results


def get_annotations(data, replay_suffix=".rep"):
    frames = []

    for item in data or []:
        anns = item.get("annotations", [])
        if not anns:
            continue

        df = pd.DataFrame(anns)

        desc = (item.get("info") or {}).get("description", "") or ""
        replay_id = None
        if isinstance(desc, str) and desc.strip():
            replay_id = desc.strip().split()[-1]

        if replay_id is None:
            replay_id = "unknown"

        if replay_suffix and not str(replay_id).endswith(replay_suffix):
            replay_id = f"{replay_id}{replay_suffix}"

        df["replay"] = replay_id
        frames.append(df)

    base_cols = [
        "replay",
        "id",
        "image_id",
        "category_id",
        "bbox",
        "area",
        "segmentation",
        "iscrowd",
    ]
    if not frames:
        return pd.DataFrame(columns=base_cols + ["score"])

    has_score = any("score" in f.columns for f in frames)

    wanted_cols = base_cols + (["score"] if has_score else [])
    for i, f in enumerate(frames):
        for col in wanted_cols:
            if col not in f.columns:
                frames[i][col] = pd.NA

    annotations = pd.concat(frames, ignore_index=True)

    return annotations[wanted_cols]


def dataframe_topk_score(dataframe, K=1):
    topk_df = (
        dataframe.sort_values("score", ascending=False)
        .groupby(["replay", "image_id"], group_keys=False)
        .head(K)
    )
    return topk_df.sort_values(["replay", "image_id"])
