
import os
import random
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image, ImageOps, ImageFilter
from .segm import SegmentationDataset


class RSVDataset(SegmentationDataset):
    NUM_CLASS = 4
    def __init__(self, cfg, stage, transform=None):
        super(RSVDataset, self).__init__(cfg, stage, transform)
        self.images, self.masks = self._get_rsv_pairs()

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.stage == 'test':
            img = img.resize((self.crop_size, self.crop_size))
            if self.transform is not None:
                img, _ = self.transform(img, None)
            return img, self.images[index]

        mask = Image.open(self.masks[index])
        if self.stage == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.stage == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.stage == 'testval'
            mask = self._mask_transform(mask)

        if self.transform is not None:
            img, mask = self.transform(img, mask)

        return img, mask, self.images[index]
    
    def __len__(self):
        return len(self.images)

    def _get_rsv_pairs(self):
        def get_path_pairs(img_folder, mask_folder):
            img_paths = []
            mask_paths = []
            for filename in os.listdir(img_folder):
                imgpath = os.path.join(img_folder, filename)
                maskname = filename.split('.')[0] + "_label.png"
                maskpath = os.path.join(mask_folder, maskname)
                if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                    img_paths.append(imgpath)
                    mask_paths.append(maskpath)
                else:
                    print("Cannot find mask or image: ", maskpath, imgpath)
            print("Found {} images in the folder {}".format(
                len(img_paths), img_folder))
            return img_paths, mask_paths

        if self.stage in ['train', 'val']:
            img_folder = os.path.join(self.root, 'images', self.stage)
            mask_folder = os.path.join(self.root, 'annotations', self.stage)
            img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
        else:
            img_folder = os.path.join(self.root, 'images', self.stage)
            img_paths = [os.path.join(img_folder, e) for e in os.listdir(img_folder)]
            mask_paths = []
        return img_paths, mask_paths

        
