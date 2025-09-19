from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Tuple, Iterable, Union

ReplayID = int


def _with_suffix(
    xs: Iterable[Union[int, str]], suffix: str = ".rep"
) -> Tuple[str, ...]:
    out = []
    for x in xs:
        s = str(x)
        out.append(s if s.endswith(suffix) else s + suffix)
    return tuple(out)


@dataclass(frozen=True)
class ExperimentSet:
    name: str
    train: Tuple[ReplayID, ...]
    test: Tuple[ReplayID, ...]
    models: Dict[str, Path]  # 예: {"vanilla": Path(...), "kbrs": Path(...)}
    suffix: str = ".rep"

    def model_vanilla(self) -> Path:
        return self.models["vanilla"]

    def model_kbrs(self) -> Path:
        return self.models["kbrs"]

    def model_legacy(self) -> Path:
        return self.models["legacy"]

    @property
    def train_replays(self) -> Tuple[str, ...]:
        return _with_suffix(self.train, self.suffix)

    @property
    def test_replays(self) -> Tuple[str, ...]:
        return _with_suffix(self.test, self.suffix)


SETS: Dict[str, ExperimentSet] = {
    "set1": ExperimentSet(
        name="set1",
        train=(36, 212, 438, 522, 1660, 1559, 1628, 2351, 6219, 11251),
        test=(275, 1725, 3613, 4520, 4664),
        models={
            "vanilla": Path("all_correct_win4_b16_20250826_045543/model_020"),
            "kbrs": Path("all_correct_win4_b16_kbrs_20250912_005403/model_020"),
        },
    ),
    "set1_prime": ExperimentSet(
        name="set1_prime",
        train=(36, 212, 438, 522, 1660, 1559, 1628, 2351, 6219, 11251),
        test=(275, 1725, 3613, 4520, 4664),
        models={
            "vanilla": Path("all_correct_win4_b16_20250823_060441/model_020"),
            "kbrs": Path("all_correct_win4_b16_kbrs_20250912_005403/model_020"),
        },
    ),
    "set2": ExperimentSet(
        name="set2",
        train=(36, 212, 438, 522, 1660, 275, 1725, 3613, 4520, 4664),
        test=(1559, 1628, 2351, 6219, 11251),
        models={
            "vanilla": Path("all_correct_win4_b16_20250823_060459/model_020"),
            "kbrs": Path("all_correct_win4_b16_kbrs_20250915_071540/model_020"),
        },
    ),
    "set3": ExperimentSet(
        name="set3",
        train=(1559, 1628, 2351, 6219, 11251, 275, 1725, 3613, 4520, 4664),
        test=(36, 212, 438, 522, 1660),
        models={
            "vanilla": Path("all_correct_win4_b16_20250823_060525/model_020"),
            "kbrs": Path("all_correct_win4_b16_kbrs_20250915_072116/model_020"),
            "legacy": Path("legacy_model_eswa/model_020"),
        },
    ),
}


def select_set(name: str) -> ExperimentSet:
    s = SETS[name]
    overlap = set(s.train) & set(s.test)
    assert not overlap, f"[{name}] train/test overlap: {sorted(overlap)}"
    return s


def override(s: ExperimentSet, **kwargs) -> ExperimentSet:
    return replace(s, **kwargs)