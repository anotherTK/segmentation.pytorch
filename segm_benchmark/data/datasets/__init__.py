
from .coco import COCODataset
from .pascal_voc import VOCDataset
from .cityscape import CityscapeDataset
from .ade20k import ADE20KDataset
from .pcontext import PContextDataset
from .concat import ConcatDataset


_DATASETS = {
    'coco': COCODataset,
    'pascal_voc': VOCDataset,
    'cityscape': CityscapeDataset,
    'ade20k': ADE20KDataset,
    'pcontext': PContextDataset,
}

def get(name, **kwargs):
    return _DATASETS[name.lower()](**kwargs)