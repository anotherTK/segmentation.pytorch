
from . import transforms as T


def build_transforms(cfg, is_train=True):
    if is_train:
        flip_horizontal_prob = cfg.DATA.HORIZON_FLIP_PROB_TRAIN
        flip_vertical_prob = cfg.DATA.VERTICAL_FLIP_PROB_TRAIN
        brightness = cfg.DATA.BRIGHTNESS
        contrast = cfg.DATA.CONTRAST
        saturation = cfg.DATA.SATURATION
        hue = cfg.DATA.HUE
    else:
        flip_horizontal_prob = 0.0
        flip_vertical_prob = 0.0
        brightness = 0.0
        contrast = 0.0
        saturation = 0.0
        hue = 0.0

    to_bgr255 = cfg.DATA.TO_BGR255
    normalize_transform = T.Normalize(
        mean=cfg.DATA.PIXEL_MEAN, 
        std=cfg.DATA.PIXEL_STD, 
        to_bgr255=to_bgr255
    )
    color_jitter = T.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )

    transform = T.Compose(
        [
            color_jitter,
            T.RandomHorizontalFlip(flip_horizontal_prob),
            T.RandomVerticalFlip(flip_vertical_prob),
            T.ToTensor(),
            normalize_transform,
        ]
    )
    return transform
