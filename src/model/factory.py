# src/model/factory.py
import os
import urllib.request

from .maskrcnn_builder import get_model_instance_segmentation
from .rtdetr_builder import get_model_instance_rtdetr

def build_model(args):
    model_name = args.model_name.lower() # "maskrcnn" or "rtdetr"
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

    else:
        raise ValueError(f"Unknown model_name: {model_name}. Use 'maskrcnn' or 'rtdetr'.")