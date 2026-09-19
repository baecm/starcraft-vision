"""One-shot diagnostic for Director-CenterNet.

Answers, with no dataset required:
  A. does the positional resnet_fpn_backbone(...) call actually work
  B. did ImageNet weights really load, and is layer1 (the FPN level the
     heatmap head reads) trainable
  C. what the heatmap resolution / effective stride actually is
  D. what amplitude the rendered targets reach, vs tau and score_threshold
  E. which parameter groups each loss term's gradient actually reaches

Run:
  docker compose -f infra/docker-compose.yml run --rm debugger debug -c \
    "PYTHONPATH=/workspace:/workspace/src python3 /workspace/scripts/probe_director.py"
"""
import sys

import torch
import torchvision
import torch.nn as nn

sys.path.insert(0, "/workspace/src")

print(f"torch {torch.__version__} | torchvision {torchvision.__version__}")
print()

# ---------------------------------------------------------------- A
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

print("[A] backbone construction")
try:
    resnet_fpn_backbone("resnet50", weights=None)
    print("    positional call            : OK")
except Exception as e:
    print(f"    positional call            : FAILED -> {type(e).__name__}: {e}")
    print("    (DirectorCenterNet's try/except would then fail again in the except branch)")

# ---------------------------------------------------------------- B
from models.backbones.director_centernet import DirectorCenterNet

model = DirectorCenterNet(in_channels=36)
body = model.backbone.body

print()
print("[B] backbone state")
try:
    from torchvision.models import ResNet50_Weights, resnet50

    ref = resnet50(weights=ResNet50_Weights.DEFAULT)
    same = torch.allclose(body.layer1[0].conv1.weight, ref.layer1[0].conv1.weight)
    print(f"    layer1 == ImageNet weights : {same}")
except Exception as e:
    print(f"    ImageNet comparison failed : {type(e).__name__}: {e}")

for name in ("conv1", "layer1", "layer2", "layer3", "layer4"):
    mod = getattr(body, name)
    tr = any(p.requires_grad for p in mod.parameters())
    print(f"    {name:<7} trainable          : {tr}")
print(f"    bn1 type                   : {type(body.bn1).__name__}")

# ---------------------------------------------------------------- C
print()
print("[C] resolution")
model.eval()
with torch.no_grad():
    feat = model._extract_features(torch.zeros(1, 36, 128, 128))
    hm, hm_logits, off, wh = model._predict_heads(feat)
print(f"    input                      : (1, 36, 128, 128)")
print(f"    fpn['0']                   : {tuple(feat.shape)}")
print(f"    heatmap                    : {tuple(hm.shape)}")
print(f"    effective stride           : {128 // hm.shape[-1]}  (model.down_ratio={model.down_ratio})")
print(f"    candidate centers          : {hm.shape[-1] * hm.shape[-2]}")

# ---------------------------------------------------------------- D
from losses.director_losses import render_gaussian_heatmap_targets

B, feat_hw = 4, hm.shape[-1]
# one 3-observer consensus mode + two 1-observer minority modes
modes = {
    "centers": torch.tensor([[64.0, 60.0], [20.0, 100.0], [100.0, 30.0]]),
    "support": torch.tensor([3.0, 1.0, 1.0]),
    "n_observers": 5,
}
rendered = render_gaussian_heatmap_targets(
    modes_list=[modes] * B,
    batch_size=B,
    feat_h=feat_hw,
    feat_w=feat_hw,
    stride=model.down_ratio,
    device=torch.device("cpu"),
)
print()
print("[D] rendered target amplitude (support 3/1/1 of U=5)")
print(f"    Y1      max                : {rendered['Y1'].max():.4f}")
print(f"    Y_minus max                : {rendered['Y_minus'].max():.4f}   <- auxiliary modes")
print(f"    model tau (conf_threshold) : {model.conf_threshold}")
print(f"    inference score_threshold  : 0.3  (conf/architecture/director_centernet.yaml)")
print(f"    |Omega| (L_rmc denominator): {rendered['mask_omega'][0].sum():.0f} of {feat_hw ** 2}")

# ---------------------------------------------------------------- E
groups = {
    "hm_head": model.hm_head,
    "off_head": model.off_head,
    "wh_head": model.wh_head,
    "backbone.fpn": model.backbone.fpn,
    "backbone.body": body,
}


def grad_norm(mod):
    total = 0.0
    for p in mod.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum())
    return total ** 0.5


torch.manual_seed(0)
model.train()
images = torch.randn(B, 36, 128, 128)
targets = [
    {"modes": modes, "next_image": torch.randn(36, 128, 128), "next_valid": True}
    for _ in range(B)
]
loss_dict = model(images, targets)

print()
print("[E] where each loss term's gradient actually lands (lambda already applied)")
hdr = f"    {'loss':<20}{'value':>10}" + "".join(f"{g:>16}" for g in groups)
print(hdr)
print("    " + "-" * (len(hdr) - 4))
for key, value in loss_dict.items():
    model.zero_grad(set_to_none=True)
    if value.requires_grad:
        value.backward(retain_graph=True)
    row = f"    {key:<20}{float(value):>10.4f}"
    for mod in groups.values():
        row += f"{grad_norm(mod):>16.3e}"
    print(row)

model.zero_grad(set_to_none=True)
total = sum(loss_dict.values())
total.backward()
print(f"    {'TOTAL':<20}{float(total):>10.4f}" + "".join(f"{grad_norm(m):>16.3e}" for m in groups.values()))
print()
print(f"    GRAD_CLIP_NORM = 2.0 is applied to this total, globally.")
print("    loss_smooth's hm_head column must be NON-zero now; it was 0 before")
print("    the soft-argmax centre, which is why scaling it down alone could not")
print("    make the objective do what it claims.")

# ---------------------------------------------------------------- F
print()
print("[F] L_smooth warmup ramp (lambda_sm x scale)")
lam = model.loss_weights["lambda_sm"]
for epoch in (0, 4, 5, 7, 10, 29):
    model.set_epoch(epoch)
    print(f"    epoch {epoch:<3} scale {model.smooth_warmup_scale():.2f}  effective weight {lam * model.smooth_warmup_scale():.4f}")
model.set_epoch(model.smooth_warmup_full)

# how many peaks survive tau at init
model.eval()
with torch.no_grad():
    boxes, scores, _, _, _ = model._extract_predicted_regions_differentiable(
        hm, off, wh, 128, 128, model.conf_threshold, model.k_max
    )
print(f"    peaks passing tau at init  : {[len(s) for s in scores]}")
