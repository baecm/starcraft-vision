# tests/test_hcm_negative_target.py
#
# hcm_negative_target selects what weights the negative term of L_hcm:
# "joint" (the auxiliary modes protected) or "primary" (CornerNet's form).
# Runs under pytest or directly (`python3 tests/test_hcm_negative_target.py`).
# Needs no data and no network: without cached ImageNet weights the backbone
# falls back to an untrained one, which is all these checks need.
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from models.backbones.director_centernet import DirectorCenterNet  # noqa: E402

HCM_ONLY = {"lambda_rmc": 0.0, "lambda_rep": 0.0, "lambda_sm": 0.0}


def _model(target: str, loss_weights=None) -> DirectorCenterNet:
    torch.manual_seed(0)
    model = DirectorCenterNet(in_channels=36, render_sigma=4.0, loss_weights=loss_weights,
                              hcm_negative_target=target)
    model.train()
    return model


def _batch(n_modes: int):
    torch.manual_seed(1)
    images = [torch.rand(36, 128, 128)]
    centers = torch.tensor([[40.0, 40.0], [90.0, 96.0], [100.0, 20.0]])[:n_modes]
    support = torch.tensor([3, 2, 1])[:n_modes]
    return images, [{"modes": {"centers": centers, "support": support, "n_observers": 5}}]


def _hcm(model, n_modes) -> torch.Tensor:
    images, targets = _batch(n_modes)
    return model(images, targets)["loss_centernet_hm"]


def test_default_is_joint():
    assert DirectorCenterNet(in_channels=36).hcm_negative_target == "joint"


def test_rejects_unknown_value():
    try:
        DirectorCenterNet(in_channels=36, hcm_negative_target="all")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_same_loss_without_auxiliary_modes():
    # With one mode Y_all == Y1, so the two forms must agree exactly.
    joint, primary = _model("joint", HCM_ONLY), _model("primary", HCM_ONLY)
    primary.load_state_dict(joint.state_dict())
    assert torch.equal(_hcm(joint, 1), _hcm(primary, 1))


def test_primary_penalizes_the_auxiliary_modes():
    # With auxiliary modes, CornerNet's form treats them as negatives, so its
    # loss is at least as large and, on an untrained net, strictly larger.
    joint, primary = _model("joint", HCM_ONLY), _model("primary", HCM_ONLY)
    primary.load_state_dict(joint.state_dict())
    lj, lp = _hcm(joint, 3), _hcm(primary, 3)
    assert lp > lj, (lj, lp)


def test_primary_drops_the_auxiliary_ignore_mask():
    # With L_rmc on, "joint" excludes the auxiliary support from L_hcm;
    # "primary" must not, so the two differ there too.
    joint, primary = _model("joint"), _model("primary")
    primary.load_state_dict(joint.state_dict())
    assert _hcm(primary, 3) > _hcm(joint, 3)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
