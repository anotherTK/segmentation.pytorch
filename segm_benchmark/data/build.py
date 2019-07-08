
import bisect
import copy
import logging
import torch.utils.data

from . import datasets as D
from . import samplers
from .transforms import build_transforms

from segm_benchmark.utils.comm import get_world_size

def build_dataset(cfg, stage, transform):

    return D.get(cfg.DATA.DATASET, cfg=cfg, stage=stage, transform=transform)


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def _quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def _compute_aspect_ratios(dataset):
    aspect_ratios = []
    for i in range(len(dataset)):
        img_info = dataset.get_img_info(i)
        aspect_ratio = float(img_info["height"]) / float(img_info["width"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def make_batch_data_sampler(
    dataset, sampler, aspect_grouping, images_per_batch, num_iters=None, start_iter=0
):
    if aspect_grouping:
        if not isinstance(aspect_grouping, (list, tuple)):
            aspect_grouping = [aspect_grouping]
        aspect_ratios = _compute_aspect_ratios(dataset)
        group_ids = _quantize(aspect_ratios, aspect_grouping)
        batch_sampler = samplers.GroupedBatchSampler(
            sampler, group_ids, images_per_batch, drop_uneven=False
        )
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_batch, drop_last=False
        )
    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler


def make_data_loader(cfg, stage, is_distributed, start_iter=0):
    num_gpus = get_world_size()
    if stage == 'train':
        images_per_gpu = cfg.SOLVER.IMS_PER_GPU
        shuffle = True
        num_iters = cfg.SOLVER.MAX_ITER
    else:
        images_per_gpu = cfg.TEST.IMS_PER_GPU
        shuffle = False if not is_distributed else True
        num_iters = None
        start_iter = 0

    aspect_grouping = [1] if cfg.DATA.ASPECT_RATIO_GROUPING else []
    transform = build_transforms(cfg, stage=='train')
    dataset = build_dataset(cfg, stage, transform)

    sampler = make_data_sampler(dataset, shuffle, is_distributed)
    batch_sampler = make_batch_data_sampler(dataset, sampler, aspect_grouping, images_per_gpu, num_iters, start_iter)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATA.NUM_WORKERS,
        batch_sampler=batch_sampler
    )

    return data_loader
