
import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from .segm import SegmentationDataset

class VOCDataset(SegmentationDataset):
    CLASSES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
        'tv/monitor', 'ambigious'
    ]
    NUM_CLASS = 21
    def __init__(self, cfg, stage, transform=None):
        # VOC2012
        super(VOCDataset, self).__init__(cfg, stage, transform)
        _mask_dir = os.path.join(self.root, 'SegmentationClass')
        _image_dir = os.path.join(self.root, 'JPEGImages')
        # train/val/test splits are pre-cut
        _splits_dir = os.path.join(self.root, 'ImageSets/Segmentation')
        if self.stage == 'train':
            _split_f = os.path.join(_splits_dir, 'trainval.txt')
        elif self.stage == 'val':
            _split_f = os.path.join(_splits_dir, 'val.txt')
        elif self.stage == 'test':
            _split_f = os.path.join(_splits_dir, 'test.txt')
        else:
            raise RuntimeError('Unknown dataset stage [{}]'.format(self.stage))

        self.images = []
        self.masks = []
        with open(os.path.join(_split_f), 'r') as lines:
            for line in tqdm(lines):
                _image = os.path.join(_image_dir, line.rstrip('\n') + '.jpg')
                assert os.path.isfile(_image)
                self.images.append(_image)
                if self.stage != 'test':
                    _mask = os.path.join(_mask_dir, line.rstrip('\n') + 'png')
                    assert os.path.isfile(_mask)
                    self.masks.append(_mask)

        if self.stage != 'test':
            assert len(self.images) == len(self.masks)

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


    def _mask_transform(self, mask):
        # TODO: reason? The mask definition?
        mask = np.array(mask).astype('int32')
        mask[mask == 255] = -1
        return torch.from_numpy(mask).long()

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0