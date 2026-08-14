# src/model/centernet_builder.py
import argparse
from .CenterNetDensityPeak import CenterNetDensityPeak

def get_model_instance_centernet(
    num_classes: int = 2,
    in_channels: int = 3,
    down_ratio: int = 4,
    max_objs: int = 100,
    use_density_peak: bool = False,
    use_kbrs: bool = False,
    kbrs_params: dict = None,
    loss_weights: dict = None
):
    """
    Factory function for CenterNet (with optional Density Peak Head & KBRS).
    """
    print(f"Building CenterNet (Density Peak Head: {use_density_peak}) - Classes: {num_classes}, Input Channels: {in_channels}")

    model = CenterNetDensityPeak(
        num_classes=num_classes,
        in_channels=in_channels,
        down_ratio=down_ratio,
        max_objs=max_objs,
        use_density_peak=use_density_peak,
        loss_weights=loss_weights
    )

    if use_kbrs:
        print("[Notice] KBRS module hook attached to CenterNet feature map.")

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--in-channels", type=int, default=3)
    args = parser.parse_args()

    model = get_model_instance_centernet(
        num_classes=args.num_classes,
        in_channels=args.in_channels
    )
    print(model)
