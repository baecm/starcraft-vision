# tests/test_mode_targets.py
#
# Mask R-CNN trained on ranked-mode boxes (mode_targets=hard/soft).
# Runs under pytest or directly (`python3 tests/test_mode_targets.py`); the
# container image has no pytest. Needs no data and no network: the backbone is
# built without pretrained weights.
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import config
from dataset.starcraft_windows import StarCraftWindowDataset
from models.backbones import maskrcnn as maskrcnn_module
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

H = W = 128
VH, VW = config.VIEWPORT_SIZE_HW


def _m_info():
    # an interior mode, one against the top-left corner, one against the
    # bottom-right corner (row, col)
    return {
        "centers": np.array([[64.0, 64.0], [2.0, 3.0], [127.0, 126.0]], dtype=np.float32),
        "support": np.array([5, 2, 1], dtype=np.int64),
        "n_observers": 5,
    }


def test_one_box_per_mode_viewport_sized_and_inside_map():
    t = StarCraftWindowDataset._make_target_from_modes(_m_info(), H, W, image_id=7)
    boxes = t["boxes"].numpy()
    assert boxes.shape == (3, 4)
    # every box is the full viewport: shifted at the edges, never cropped
    assert np.allclose(boxes[:, 2] - boxes[:, 0], VW)
    assert np.allclose(boxes[:, 3] - boxes[:, 1], VH)
    assert boxes.min() >= 0 and boxes[:, 2].max() <= W and boxes[:, 3].max() <= H
    # the interior mode is centered on its center
    assert np.allclose(boxes[0], [64 - VW / 2, 64 - VH / 2, 64 + VW / 2, 64 + VH / 2])
    assert np.allclose(boxes[1, :2], [0, 0])
    assert np.allclose(boxes[2, 2:], [W, H])
    # masks are the boxes
    for b, m in zip(boxes.astype(int), t["masks"].numpy()):
        assert m.sum() == VH * VW
        assert m[b[1]:b[3], b[0]:b[2]].all()
    assert t["labels"].tolist() == [1, 1, 1]
    assert np.allclose(t["mode_weight"].numpy(), [1.0, 0.4, 0.2])


def test_frame_without_modes_gives_empty_target():
    t = StarCraftWindowDataset._make_target_from_modes(None, H, W, image_id=7)
    assert t["boxes"].shape == (0, 4)
    assert t["masks"].shape == (0, H, W)
    assert t["mode_weight"].numel() == 0


def _build(soft: bool):
    # no pretrained weights: the test must run offline
    orig = maskrcnn_module.resnet_fpn_backbone
    maskrcnn_module.resnet_fpn_backbone = lambda backbone_name, weights=None, **kw: resnet_fpn_backbone(
        backbone_name=backbone_name, weights=None, **kw)
    try:
        torch.manual_seed(0)
        model = maskrcnn_module.build_maskrcnn_backbone(
            num_classes=2, in_channels=36, min_sizes=(256,), max_size=256, soft_mode_cls=soft)
    finally:
        maskrcnn_module.resnet_fpn_backbone = orig
    return model


def _batch():
    images = [torch.rand(36, H, W), torch.rand(36, H, W)]
    targets = [
        StarCraftWindowDataset._make_target_from_modes(_m_info(), H, W, image_id=1),
        StarCraftWindowDataset._make_target_from_modes(None, H, W, image_id=2),
    ]
    return images, targets


def test_soft_heads_replace_only_the_classifier_loss():
    model = _build(soft=True)
    assert isinstance(model.roi_heads, maskrcnn_module.SupportTargetRoIHeads)
    model.train()
    images, targets = _batch()
    losses = model(images, targets)
    assert set(losses) == {"loss_classifier", "loss_box_reg", "loss_mask", "loss_objectness", "loss_rpn_box_reg"}
    assert all(torch.isfinite(v) for v in losses.values())
    sum(losses.values()).backward()
    assert model.roi_heads.box_predictor.cls_score.weight.grad is not None

    # the soft targets are support / U on positives, 0 on negatives
    heads = model.roi_heads
    seen = {}

    def spy(proposals, tgts, _orig=heads.select_training_samples):
        out = _orig(proposals, tgts)
        seen["labels"] = torch.cat(out[2])
        seen["q"] = heads._fg_targets.clone()
        return out

    heads.select_training_samples = spy
    model(images, targets)
    labels, q = seen["labels"], seen["q"]
    assert (q[labels == 0] == 0).all()
    pos = q[labels > 0]
    assert pos.numel() > 0  # gt boxes are added as proposals
    allowed = np.array([1.0, 0.4, 0.2])
    assert all(np.isclose(allowed, v, atol=1e-6).any() for v in pos.numpy())


def test_soft_loss_equals_one_hot_loss_when_every_weight_is_one():
    images, targets = _batch()
    for t in targets:
        t["mode_weight"] = torch.ones_like(t["mode_weight"])
    hard, soft = _build(soft=False), _build(soft=True)
    soft.load_state_dict(hard.state_dict())
    hard.train(), soft.train()
    torch.manual_seed(1)
    l_hard = hard(images, targets)
    torch.manual_seed(1)
    l_soft = soft(images, targets)
    assert torch.allclose(l_hard["loss_classifier"], l_soft["loss_classifier"], atol=1e-6)


def test_soft_model_infers_like_a_plain_one():
    hard, soft = _build(soft=False), _build(soft=True)
    soft.load_state_dict(hard.state_dict())
    hard.eval(), soft.eval()
    images, _ = _batch()
    with torch.no_grad():
        a, b = hard(images), soft(images)
    for x, y in zip(a, b):
        assert torch.equal(x["boxes"], y["boxes"]) and torch.equal(x["scores"], y["scores"])
    assert soft.roi_heads._class_logits is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
