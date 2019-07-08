
import random
import numpy as np
import torch

from PIL import Image, ImageOps, ImageFilter

class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, stage, transform=None):
        self.root = cfg.DATA.ROOT
        self.transform = transform
        self.base_size = cfg.DATA.BASE_SIZE
        self.crop_size = cfg.DATA.CROP_SIZE
        self.rot_range = cfg.DATA.ROT_RANGE
        self.stage = stage
        self.cfg = cfg

    def __getitem__(self, index):
        raise NotImplemented

    @property
    def num_class(self):
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        raise NotImplemented

    def make_pred(self, mask):
        return mask + self.pred_offset

    def _val_sync_transform(self, img, mask):
        # img: PIL image
        crop_size = self.crop_size
        w, h = img.size
        # resize the short size to crop_size
        if w > h:
            oh = crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - crop_size) / 2.0))
        y1 = int(round((h - crop_size) / 2.0))
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # transform
        return img, self._mask_transform(mask)

    def _sync_transform(self, img, mask):
        # img: PIL image
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        # random rotate
        rot = random.uniform(-self.rot_range, self.rot_range)
        img = img.rotate(rot, resample=Image.BILINEAR)
        mask = mask.rotate(rot, resample=Image.NEAREST)

        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)

        # random crop
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))

        # gaussian blur
        if random.random() < 0.5:
            # TODO: radius < 1, does this help?
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))

        # transform
        return img, self._mask_transform(mask)

    def _mask_transform(self, mask):
        return torch.from_numpy(np.array(mask)).long()