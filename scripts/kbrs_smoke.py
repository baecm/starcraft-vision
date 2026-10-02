#!/usr/bin/env python3
"""
kbrs_smoke.py
=============

Build the Mask R-CNN + KBRS model exactly as train.py would from the Hydra
config, run one training forward pass on a small synthetic batch, and check
what the auxiliary loss actually does before committing GPU-days to it:

    - the component weights the scorer received (a run once trained at 1/1/1
      because the weights sat under a key the model did not read)
    - the projection and gate channels after expansion over the window
    - that density, centeredness and mixture are all non-zero
    - that loss_kbrs carries a gradient, and that the gradient reaches the
      backbone (an earlier wrapper scored the raw input and reached nothing)

    make kbrs-smoke
    make kbrs-smoke ARGS="plugins/kbrs/score=mixture/000"   # any Hydra override

Exits non-zero if any check fails.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

sys.path.insert(0, "/workspace/src")
from models.factory import build_model  # noqa: E402


def main() -> int:
    overrides = ["plugins/kbrs=enabled"] + sys.argv[1:]
    with initialize_config_dir(config_dir="/workspace/conf", version_base=None):
        cfg = compose(config_name="config", overrides=overrides)

    kbrs_params = OmegaConf.to_container(cfg.kbrs.kbrs_params, resolve=True)
    loss_weights = {k: float(v) for k, v in (OmegaConf.to_container(cfg.kbrs.loss_weights, resolve=True) or {}).items()}
    window = int(cfg.window_size)
    in_channels = 9 * window
    args = SimpleNamespace(
        model_name=cfg.architecture.model_name, use_kbrs=True, num_classes=2,
        in_channels=in_channels, window_size=window, kbrs_params=kbrs_params,
        loss_weights=loss_weights, resize_mode=getattr(cfg.architecture, "resize_mode", "resize"),
        do_normalize=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Seed before building: the FPN initialisation decides the sign of the
    # feature sums, and with it whether the unrectified mixture is zero.
    torch.manual_seed(int(cfg.seed))
    model = build_model(args).to(device).train()
    hook = model.kbrs_hook

    print(f"device               : {device}")
    print(f"score weights used   : {hook.weights}")
    print(f"projections (expanded): {hook.scorer.projections}")
    print(f"gate channels        : {hook.gate_channels} gain={hook.gate_gain} reduce={hook.gate_reduce}")
    print(f"tau / detach / fmap  : {hook.tau} / {hook.detach_scorer_input} / {hook.feature_map_name}")
    print(f"region (kh, kw)      : {hook.region_size}")
    print(f"mixture_nonneg       : {hook.scorer.mixture_nonneg}")
    print(f"loss_kbrs weight     : {loss_weights.get('loss_kbrs', 0.25)}")

    # Sparse non-negative counts, like the channelised game state.
    images = [(torch.rand(in_channels, 128, 128) > 0.97).float().to(device) for _ in range(2)]
    targets = []
    for _ in images:
        mask = torch.zeros(1, 128, 128, dtype=torch.uint8)
        mask[0, 40:52, 30:50] = 1
        targets.append({
            "boxes": torch.tensor([[30.0, 40.0, 50.0, 52.0]], device=device),
            "labels": torch.tensor([1], device=device),
            "masks": mask.to(device),
        })

    losses = model(images, targets)
    print("losses               : " + ", ".join(f"{k}={float(v.detach()):.4f}" for k, v in losses.items()))

    failures = []
    comps = hook.last_components
    for name in ("density", "centeredness", "mixture"):
        if hook.weights.get(name, 0.0) == 0.0:
            print(f"{name:<21}: weight 0, not computed")
            continue
        t = comps.get(name)
        if t is None:
            failures.append(f"{name} missing")
            continue
        print(f"{name:<21}: shape={tuple(t.shape)} mean={t.mean():.4g} std={t.std():.4g} max={t.abs().max():.4g}")
        if float(t.abs().max()) == 0.0:
            failures.append(f"{name} is identically zero")

    lk = losses["loss_kbrs"]
    print(f"loss_kbrs requires_grad: {lk.requires_grad}")
    if not lk.requires_grad:
        failures.append("loss_kbrs has no gradient")
    else:
        model.zero_grad(set_to_none=True)
        lk.backward()
        backbone = model.base_model.backbone
        norms = {
            "fpn": sum(float(p.grad.norm()) for p in backbone.fpn.parameters() if p.grad is not None),
            "body.layer4": sum(float(p.grad.norm()) for p in backbone.body.layer4.parameters() if p.grad is not None),
            "roi_heads": sum(float(p.grad.norm()) for p in model.base_model.roi_heads.parameters() if p.grad is not None),
        }
        print("grad norm from loss_kbrs alone: " + ", ".join(f"{k}={v:.3e}" for k, v in norms.items()))
        if norms["fpn"] == 0.0:
            failures.append("loss_kbrs does not reach the FPN")
        if norms["roi_heads"] != 0.0:
            failures.append("loss_kbrs reaches the ROI heads, which it should not")

    if failures:
        print("FAIL: " + "; ".join(failures))
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
