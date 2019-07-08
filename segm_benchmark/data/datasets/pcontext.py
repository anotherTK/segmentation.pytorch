
import os
import numpy as np
import torch
from PIL import Image
from tqdm import trange
from .segm import SegmentationDataset
from detail import Detail

class PContextDataset(SegmentationDataset):
    NUM_CLASS = 59
    def __init__(self, cfg, stage, transform=None):
        # VOC2010
        super(PContextDataset, self).__init__(cfg, stage, transform)

        annFile = os.path.join(self.root, 'trainval_merged.json')
        imgDir = os.path.join(self.root, 'JPEGImages')

        self.detail = Detail(annFile, imgDir, self.stage)
        self.ids = self.detail.getImgs()
        self._mapping = np.sort(np.array([
            0, 2, 259, 260, 415, 324, 9, 258, 144, 18, 19, 22,
            23, 397, 25, 284, 158, 159, 416, 33, 162, 420, 454, 295, 296, 427, 44, 45, 46, 308, 59, 440, 445,31, 232, 65, 354, 424, 68, 326, 72, 458, 34, 207, 80, 355, 85, 347, 220, 349, 360, 
            98, 187, 104, 105, 366, 189, 368, 113, 115
        ]))
        self._key = np.array(range(len(self._mapping))).astype('uint8')
        mask_file = os.path.join(self.root, self.stage + '.pth')
        if os.path.exists(mask_file):
            self.masks = torch.load(mask_file)
        else:
            self.masks = self._preprocess(mask_file)

    def _class_to_index(self, mask):
        values = np.unique(mask)
        for i in range(len(values)):
            assert values[i] in self._mapping
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        return self._key[index].reshape(mask.shape)


    def _preprocess(self, mask_file):
        masks = {}
        tbar = trange(len(self.ids))
        for i in tbar:
            img_id = self.ids[i]
            mask = Image.fromarray(self._class_to_index(self.detail.getMask(img_id)))
            masks[img_id['image_id']] = mask
            tbar.set_description("Preprocessing masks {}".format(img_id['image_id']))
        torch.save(masks, mask_file)
        return masks

    def _mask_transform(self, mask):
        mask = np.array(mask).astype('int32') - 1
        return torch.from_numpy(mask).long()

    def __getitem__(self, index):
        img_id = self.ids[index]
        path = img_id['file_name']
        iid = img_id['image_id']
        img = Image.open(os.path.join(self.detail.img_foler, path)).convert('RGB')
        
        if self.stage == 'test':
            if self.transform is not None:
                img, _ = self.transform(img, None)
            return img, None

        mask = self.masks[iid]
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

    @property
    def pred_offset(self):
        return 1
    
