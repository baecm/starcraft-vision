# src/pipeline.py
from __future__ import annotations

from types import SimpleNamespace

import hydra
from omegaconf import DictConfig

from train import run_training
from inference import run_inference


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # 1) train 실행
    if cfg.mode in ("train_only", "train_and_inference"):
        run_training(cfg)

    # 2) cfg를 이용해서 inference용 args 조립
    if cfg.mode in ("inference_only", "train_and_inference"):
        inf_args = SimpleNamespace(
            # inference.py의 parse_inference_args()에서 정의한 것과 동일하게:
            replays=list(cfg.test_replays),
            model_root=cfg.log_root,
            model_name=cfg.id_string,       # train에서 최종적으로 설정된 id_string 재사용
            model_number=str(cfg.max_epoch),
            data_root=cfg.data_root,
            label_method=cfg.label_method,
            window_size=cfg.window_size,
            sample_ratio=cfg.sample_ratio,
            seed=cfg.seed,
            cuda=cfg.cuda,
            workers=None,                   # 필요하면 cfg.inference.workers 같은 걸 두고 채우기
            score_threshold=None,           # 마찬가지로 cfg.inference.score_threshold 등
            out_dir=cfg.inference.out_dir if "inference" in cfg else "./results",
            # parse_inference_args에서 쓰는 나머지 인자들도 여기에서 채워주면 됨
        )

        # 3) inference 실행
        run_inference(inf_args)


if __name__ == "__main__":
    main()