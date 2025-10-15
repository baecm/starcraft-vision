# src/model/maskrcnn_builder.py
import argparse
import torch

import torch.nn as nn
import torchvision

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from .CustomRCNNTransform import CustomRCNNTransform
from .KBRS_MaskRCNN import KBRS_MaskRCNN


def _kaiming_init_conv(conv: nn.Conv2d):
    nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


def get_model_instance_segmentation(num_classes: int,
                                    window_size: int = 1,
                                    in_channels: int = None,
                                    do_normalize: bool = False,
                                    normalize_mean=None,
                                    normalize_std=None,
                                    resize_mode: str = "resize",
                                    min_sizes=None,
                                    max_size: int = 1333,
                                    rpn_small_anchors: bool = False,
                                    use_kbrs: bool = False,
                                    kbrs_params=None,
                                    loss_weights=None):
    """
    - in_channels 미지정 시 9 * window_size
    - 첫 conv를 in_channels로 교체 + Kaiming init
    - resize_mode / min_sizes / max_size로 리사이즈 제어
    - do_normalize=True면 mean/std 사용(길이는 in_channels와 동일해야 함)
    - rpn_small_anchors=True면 작은 입력용 앵커로 교체
    """
    if in_channels is None:
        in_channels = 9 * window_size
    print(f"Using {in_channels} input channels (window size: {window_size})")

    if use_kbrs:
        # 자동 확장 메타 주입
        kbrs_params = dict(kbrs_params or {})
        kbrs_params["window_size"] = int(window_size)
        kbrs_params.setdefault("per_window", 9)

        backbone = resnet_fpn_backbone("resnet50", weights="DEFAULT")
        backbone.body.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        _kaiming_init_conv(backbone.body.conv1)

        model = KBRS_MaskRCNN(
            backbone, num_classes,
            kbrs_params=kbrs_params,
            loss_weights=loss_weights
        )
    else:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
        model.backbone.body.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        _kaiming_init_conv(model.backbone.body.conv1)

    # heads
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    # 작은 앵커 (리사이즈 끌 때 유리)
    if rpn_small_anchors:
        from torchvision.models.detection.rpn import AnchorGenerator
        anchor_generator = AnchorGenerator(
            sizes=((8,), (16,), (32,), (64,), (128,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5
        )
        model.rpn.anchor_generator = anchor_generator

    # --- Transform 설정 ---
    C = in_channels
    if do_normalize:
        image_mean = (normalize_mean if normalize_mean is not None else [0.0] * C)
        image_std  = (normalize_std  if normalize_std  is not None else [1.0] * C)
    else:
        image_mean = [0.0] * C
        image_std  = [1.0] * C

    if min_sizes is None:
        min_sizes = [800] if resize_mode == "resize" else [128]

    # resize_mode="keep"이면 resize를 스킵
    do_resize = (resize_mode == "resize")

    model.transform = CustomRCNNTransform(
        min_size=min_sizes,
        max_size=max_size,
        image_mean=image_mean,
        image_std=image_std,
        do_normalize=do_normalize,
        do_resize=do_resize
    )

    assert len(model.transform.image_mean) == in_channels
    assert len(model.transform.image_std) == in_channels
    return model


if __name__ == "__main__":

    def count_params(m):
        total = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        return total, trainable

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--in-channels", type=int, default=9)
    parser.add_argument("--do-normalize", action="store_true")
    parser.add_argument("--normalize-mean", type=float, nargs="*", default=None)
    parser.add_argument("--normalize-std", type=float, nargs="*", default=None)
    parser.add_argument("--resize-mode", choices=["resize", "keep"], default="resize")
    parser.add_argument("--min-sizes", type=int, nargs="*", default=None)
    parser.add_argument("--max-size", type=int, default=1333)
    parser.add_argument("--rpn-small-anchors", action="store_true")
    parser.add_argument("--use-kbrs", action="store_true")
    args = parser.parse_args()

    model = get_model_instance_segmentation(
        num_classes=args.num_classes,
        window_size=args.window_size,
        in_channels=args.in_channels,
        do_normalize=args.do_normalize,
        normalize_mean=args.normalize_mean,
        normalize_std=args.normalize_std,
        resize_mode=args.resize_mode,
        min_sizes=args.min_sizes,
        max_size=args.max_size,
        rpn_small_anchors=args.rpn_small_anchors,
        use_kbrs=args.use_kbrs,
        kbrs_params=None,
        loss_weights=None,
    )

    print("\n===== Model Structure =====")
    print(model)

    print("\n===== Transform Config =====")
    T = model.transform
    print(f"do_resize: {getattr(T, 'do_resize', True)}")
    print(f"min_size: {getattr(T, 'min_size', None)}")
    print(f"max_size: {getattr(T, 'max_size', None)}")
    print(f"do_normalize: {getattr(T, 'do_normalize', True)}")
    print(f"image_mean(len={len(T.image_mean)}): [first 5] {T.image_mean[:5] if hasattr(T,'image_mean') else None}")
    print(f"image_std (len={len(T.image_std)}): [first 5] {T.image_std[:5] if hasattr(T,'image_std') else None}")

    print("\n===== RPN / Anchors =====")
    ag = model.rpn.anchor_generator
    try:
        sizes = tuple(tuple(int(s) for s in sz) for sz in ag.sizes)
    except Exception:
        sizes = ag.sizes
    print(f"anchor sizes: {sizes}")
    print(f"anchor aspect_ratios: {ag.aspect_ratios}")

    print("\n===== ROI Heads =====")
    print(f"box_predictor in_features: {model.roi_heads.box_predictor.cls_score.in_features}")
    print(f"mask_predictor in_channels: {model.roi_heads.mask_predictor.conv5_mask.in_channels}")

    total, trainable = count_params(model)
    print("\n===== Parameters =====")
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")

    try:
        from torchinfo import summary as torchinfo_summary
        c_in = model.backbone.body.conv1.in_channels
        print("\n===== torchinfo.summary (dummy input 1xCx512x512) =====")
        torchinfo_summary(model, input_size=(1, c_in, 512, 512), verbose=0,
                          col_names=("input_size","output_size","num_params","kernel_size","mult_adds"))
    except Exception as e:
        print("\n(torchinfo 미설치 또는 실행 생략:", str(e), ")")
