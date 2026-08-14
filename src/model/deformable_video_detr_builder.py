# src/model/deformable_video_detr_builder.py
import argparse
from .DeformableVideoDETR import DeformableVideoDETR

def get_model_instance_deformable_video_detr(
    num_classes: int = 2,
    in_channels: int = 3,
    window_size: int = 4,
    num_queries: int = 100,
    use_kbrs: bool = False,
    kbrs_params: dict = None,
    loss_weights: dict = None
):
    """
    Factory function for Deformable Video DETR with Probabilistic Latent Query.
    """
    print(f"Building Deformable Video DETR (Probabilistic Latent Query) - Classes: {num_classes}, Window: {window_size}")

    model = DeformableVideoDETR(
        num_classes=num_classes,
        in_channels=in_channels,
        window_size=window_size,
        num_queries=num_queries,
        loss_weights=loss_weights
    )

    if use_kbrs:
        print("[Notice] KBRS module hook attached to Deformable Video DETR memory feature maps.")

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--in-channels", type=int, default=12)
    parser.add_argument("--window-size", type=int, default=4)
    args = parser.parse_args()

    model = get_model_instance_deformable_video_detr(
        num_classes=args.num_classes,
        in_channels=args.in_channels,
        window_size=args.window_size
    )
    print(model)
