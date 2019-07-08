""" COCO categories definition
NUM_CHANNEL = 91
################################
[] background
[5] airplane
[2] bicycle
[16] bird
[9] boat
[44] bottle
[6] bus
[3] car
[17] cat
[62] chair
[21] cow
[67] dining table
[18] dog
[19] horse
[4] motorcycle
[1] person
[64] potted plant
[20] sheep
[63] couch
[7] train
[72] tv
###############################
"""

import os
import numpy as np
import torch
from PIL import Image
from tqdm import trange
from .segm import SegmentationDataset
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask

class COCODataset(SegmentationDataset):
    NUM_CLASS = 21
    CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72]
    
    def __init__(self, cfg, stage, transform=None):
        super(COCODataset, self).__init__(cfg, stage, transform)
        if self.stage == 'train':
            ann_file = os.path.join(self.root, 'annotations/instances_train2017.json')
            ids_file = os.path.join(self.root, 'annotations/train_ids.pth')
            # update root to images directory
            self.root = os.path.join(self.root, 'train2017')
        else:
            ann_file = os.path.join(self.root, 'annotations/instances_val2017.json')
            ids_file = os.path.join(self.root, 'annotations/val_ids.pth')
            self.root = os.path.join(self.root, 'val2017')

        self.coco = COCO(ann_file)
        
        if os.path.exists(ids_file):
            self.ids = torch.load(ids_file)
        else:
            ids = list(self.coco.imgs.keys())
            self.ids = self._preprocess(ids, ids_file)

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_meta = self.coco.loadImgs(img_id)[0]
        filename = img_meta['file_name']
        img = Image.open(os.path.join(self.root, filename)).convert('RGB')
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        mask = Image.fromarray(self._generate_mask(anns, img_meta['height'], img_meta['width']))

        # synchronized transform
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
        return len(self.ids)
        
    def _preprocess(self, ids, ids_file):
        tbar = trange(len(ids))
        new_ids = []
        for i in tbar:
            img_id = ids[i]
            anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            img_meta = self.coco.loadImgs(img_id)[0]
            mask = self._generate_mask(anns, img_meta['height'], img_meta['width'])

            # filter by, more than 1000 pixel
            if (mask > 0).sum() > 1000:
                new_ids.append(img_id)
            tbar.set_description("Processing {}/{}, got {} qualified images".format(i, len(ids), len(new_ids)))

        torch.save(new_ids, ids_file)
        print("Preprocess finished, qualified ids have been saved to {}".format(ids_file))
        return new_ids

    def _generate_mask(self, anns, h, w):
        mask = np.zeros((h ,w), dtype=np.uint8)
        for ann in anns:
            rle = coco_mask.frPyObjects(ann['segmentation'], h, w)
            m = coco_mask.decode(rle)
            cat = ann['category_id']
            if cat in self.CAT_LIST:
                c = self.CAT_LIST.index(cat)
                if len(m.shape) < 3:
                    mask[:, :] += (mask == 0) * (m * c)
                else:
                    mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)

        return mask
