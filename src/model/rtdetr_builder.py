# src/model/rtdetr_builder.py
from .factory import build_model
from types import SimpleNamespace

def get_model_instance_rtdetr(**kwargs):
    kwargs["model_name"] = "rtdetr"
    return build_model(SimpleNamespace(**kwargs))