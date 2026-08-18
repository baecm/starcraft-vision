# src/model/centernet_builder.py
from .factory import build_model
from types import SimpleNamespace

def get_model_instance_centernet(**kwargs):
    kwargs["model_name"] = "centernet"
    return build_model(SimpleNamespace(**kwargs))
