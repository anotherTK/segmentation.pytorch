
import argparse
import os

import torch
from segm_benchmark.config import cfg
from segm_benchmark.data import make_data_loader
from segm_benchmark.engine.inference import inference
from segm_benchmark.models import build_model
from segm_benchmark.utils.checkpoint import Checkpointer
from segm_benchmark.utils.comm import synchronize, get_rank
from segm_benchmark.utils.logger import setup_logger
from segm_benchmark.utils.miscellaneous import mkdir

# Check if we can enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for mixed precision via apex.amp')


def main():
    parser = argparse.ArgumentParser(
        description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]
                   ) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("segm_benchmark", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    model = build_model(cfg, jpu=cfg.MODEL.JPU, lateral=cfg.MODEL.LATERAL)
    model.to(cfg.MODEL.DEVICE)

    # Initialize mixed-precision if necessary
    use_mixed_precision = cfg.DTYPE == 'float16'
    amp_handle = amp.init(enabled=use_mixed_precision)
    amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    model = amp.initialize(model, opt_level=amp_opt_level)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    output_dir = cfg.OUTPUT_DIR
    checkpointer = Checkpointer(model, save_dir=output_dir)
    ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
    _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)
    dataset_name = cfg.DATA.DATASET
    if cfg.OUTPUT_DIR:
        output_folder = os.path.join(
            cfg.OUTPUT_DIR, "inference", dataset_name)
        mkdir(output_folder)
    data_loader_val = make_data_loader(
        cfg, stage='test', is_distributed=distributed)
    inference(
        model,
        data_loader_val,
        dataset_name=dataset_name,
        device=cfg.MODEL.DEVICE,
        output_folder=output_folder,
    )
    synchronize()


if __name__ == "__main__":
    main()
