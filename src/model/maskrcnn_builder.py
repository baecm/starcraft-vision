# src/model/maskrcnn_builder.py
from .factory import build_model
from types import SimpleNamespace

def get_model_instance_segmentation(**kwargs):
    kwargs["model_name"] = "maskrcnn"
    return build_model(SimpleNamespace(**kwargs))
