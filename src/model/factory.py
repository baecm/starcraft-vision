# src/model/factory.py
import os
import urllib.request

from .maskrcnn_builder import get_model_instance_segmentation
from .rtdetr_builder import get_model_instance_rtdetr
from .centernet_builder import get_model_instance_centernet
from .deformable_video_detr_builder import get_model_instance_deformable_video_detr

def build_model(args):
    model_name = args.model_name.lower() # "maskrcnn", "rtdetr", "centernet", "deformable_video_detr"
    use_kbrs = args.use_kbrs

    print(f"========== Building Model ==========")
    print(f"Architecture : {model_name.upper()}")
    print(f"Use KBRS     : {use_kbrs}")
    
    if model_name == "maskrcnn":
        model = get_model_instance_segmentation(
            num_classes=args.num_classes,
            window_size=args.window_size,
            in_channels=args.in_channels,
            use_kbrs=use_kbrs,
            kbrs_params=args.kbrs_params if use_kbrs else None,
            loss_weights=args.loss_weights
        )
        return model

    elif model_name == "rtdetr":
        model = get_model_instance_rtdetr(
            num_classes=args.num_classes,
            version=args.rtdetr_version,
            model_size=args.rtdetr_size,
            in_channels=args.in_channels,
            use_kbrs=use_kbrs,
            kbrs_params=args.kbrs_params if use_kbrs else None,
            loss_weights=args.loss_weights
        )
        return model

    elif model_name in ["centernet", "centernet_density_peak"]:
        use_dp = getattr(args, "use_density_peak", model_name == "centernet_density_peak")
        model = get_model_instance_centernet(
            num_classes=args.num_classes,
            in_channels=args.in_channels,
            down_ratio=getattr(args, "centernet_down_ratio", 4),
            max_objs=getattr(args, "max_objs", 100),
            use_density_peak=use_dp,
            use_kbrs=use_kbrs,
            kbrs_params=args.kbrs_params if use_kbrs else None,
            loss_weights=args.loss_weights
        )
        return model

    elif model_name in ["deformable_video_detr", "deformable_detr", "deformable_video_detr_probabilistic"]:
        use_pq = getattr(args, "use_probabilistic_query", "probabilistic" in model_name)
        model = get_model_instance_deformable_video_detr(
            num_classes=args.num_classes,
            in_channels=args.in_channels,
            window_size=args.window_size,
            num_queries=getattr(args, "num_queries", 100),
            use_probabilistic_query=use_pq,
            use_kbrs=use_kbrs,
            kbrs_params=args.kbrs_params if use_kbrs else None,
            loss_weights=args.loss_weights
        )
        return model

    else:
        raise ValueError(f"Unknown model_name: {model_name}. Use 'maskrcnn', 'rtdetr', 'centernet', or 'deformable_video_detr'.")