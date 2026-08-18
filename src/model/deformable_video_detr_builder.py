# src/model/deformable_video_detr_builder.py
from .factory import build_model
from types import SimpleNamespace

def get_model_instance_deformable_video_detr(**kwargs):
    kwargs["model_name"] = "deformable_video_detr"
    return build_model(SimpleNamespace(**kwargs))
