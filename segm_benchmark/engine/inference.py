
import logging
import time
import os

import torch
from tqdm import tqdm
import numpy as np

from segm_benchmark.config import cfg
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
from segm_benchmark.utils.metrics import batch_intersection_union, batch_pix_accuracy

def _evaluate(pred, target, nclass):
    correct, labeled = batch_pix_accuracy(pred, target)
    inter, union = batch_intersection_union(pred, target, nclass)
    return correct, labeled, inter, union


def compute_on_dataset(model, data_loader, device, timer=None):
    model.eval()
    results_dict = {}
    results_metr = []
    cpu_device = torch.device("cpu")
    for _, batch in enumerate(tqdm(data_loader)):
        targets = None
        if len(batch) == 3:
            images, targets, image_ids = batch
        else:
            images, image_ids = batch
        with torch.no_grad():
            if timer:
                timer.tic()
            outputs = model(images.to(device))
            if targets is not None:
                metr = _evaluate(outputs, targets.to(device), model.nclass)
                results_metr.append(metr)
            if timer:
                if not cfg.MODEL.DEVICE == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
            outputs = [o.to(cpu_device) for o in outputs]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, outputs)}
        )
    return results_dict, results_metr


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return

    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)

    image_ids = list(sorted(predictions.keys()))
    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def _accumulate_metrs_from_multiple_gpus(metrs_per_gpu):
    all_predictions = all_gather(metrs_per_gpu)
    if not is_main_process():
        return
    # merge the list of list
    predictions = None
    if all_predictions is not None and all_predictions[0] is not None:
        total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        for item in all_predictions:
            for p in item:
                total_correct += p[0]
                total_label += p[1]
                total_inter += p[2]
                total_union += p[3]
        
        pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
        IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
        mIoU = IoU.mean()
        predictions = {
            "pixAcc": pixAcc,
            "mIOU": mIoU
        }
        
    return predictions

def inference(
        model,
        data_loader,
        dataset_name,
        device="cuda",
        output_folder=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("segm_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(
        dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions, metrs = compute_on_dataset(
        model, data_loader, device, inference_timer)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time *
            num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    metrs  = _accumulate_metrs_from_multiple_gpus(metrs)
    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    if metrs is not None:
        print(metrs)

