
from .fcn import FCN
from .encnet import EncNet

from segm_benchmark.data.datasets import _DATASETS

_MODELS = {
    "fcn": FCN,
    "encnet": EncNet
}

def build_model(cfg, **kwargs):
    model = _MODELS[cfg.MODEL.NET](_DATASETS[cfg.DATA.DATASET].NUM_CLASS, backbone=cfg.MODEL.BACKBONE, **kwargs)

    return model
