
import os
import random
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image, ImageOps, ImageFilter
from .segm import SegmentationDataset


class CityscapeDataset(SegmentationDataset):
    NUM_CLASS = 19
    def __init__(self, cfg, stage, transform=None):
        super(CityscapeDataset, self).__init__(cfg, stage, transform)
        self.images, self.masks = self._get_cityscape_pairs()
        assert len(self.images) == len(self.masks)
        
        self._indices = np.array(range(-1, NUM_CLASS))
        self._classes = np.array([0, 7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33])
        self._key = np.array([-1, -1, -1, -1, -1, -1,
                              -1, -1,  0,  1, -1, -1, 
                              2,   3,  4, -1, -1, -1,
                              5,  -1,  6,  7,  8,  9,
                              10, 11, 12, 13, 14, 15,
                              -1, -1, 16, 17, 18])
        self._mapping = np.array(range(-1, len(self._key) - 1)).astype('int32')

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.stage == 'test':
            if self.transform is not None:
                img, _ = self.transform(img, None)
            return img, None

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

        return img, mask

    def __len__(self):
        return len(self.images)

    def make_pred(self, mask):
        values = np.unique(mask)
        for i in range(len(values)):
            assert values[i] in self._indices
        index = np.digitize(mask.ravel(), self._indices, right=True)
        return self._classes[index].reshape(mask.shape)

    def _mask_transform(self, mask):
        mask = self._class_to_index(np.array(mask).astype('int32'))
        return torch.from_numpy(mask).long()

    def _class_to_index(self, mask):
        values = np.unique(mask)
        for i in range(len(values)):
            assert values[i] in self._mapping
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        return self._key[index].reshape(mask.shape)


    def _get_cityscape_pairs(self):
        def get_path_pairs(img_folder, mask_folder):
            img_paths = []
            mask_paths = []
            for root, directories, files in os.walk(img_foler):
                for filename in files:
                    imgpath = os.path.join(root, filename)
                    foldername = os.path.basename(os.path.dirname(imgpath))
                    maskname = filename.replace('leftImg8bit', 'gtFine_labelIds')
                    maskpath = os.path.join(mask_folder, foldername, maskname)
                    if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                        img_paths.append(imgpath)
                        mask_paths.append(maskpath)
                    else:
                        print("Cannot find mask or image: ", maskpath, imgpath)
            print("Found {} images in the folder {}".format(len(img_paths), img_folder))
            return img_paths, mask_paths

        if self.stage in ['train', 'val', 'test']:
            img_folder = os.path.join(self.root, 'leftImg8bit', self.stage)
            mask_folder = os.path.join(self.root, 'gtFine', self.stage)
            img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
            return img_paths, mask_paths
        else:
            assert self.stage == 'trainval'
            img_paths = []
            mask_paths = []
            for stage in ['train', 'val']:
                img_folder = os.path.join(self.root, 'leftImg8bit', stage)
                mask_folder = os.path.join(self.root, 'gtFine', stage)
                _img_paths, _mask_paths = get_path_pairs(img_folder, mask_folder)
                img_paths += _img_paths
                mask_paths += _mask_paths

        return img_paths, mask_paths
            

