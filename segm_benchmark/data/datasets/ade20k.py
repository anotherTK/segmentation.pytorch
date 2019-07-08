
import os
import numpy as np
import torch
from PIL import Image
from .segm import SegmentationDataset

class ADE20KDataset(SegmentationDataset):
    NUM_CLASS = 150
    def __init__(self, cfg, stage, transform=None):
        super(ADEDataset, self).__init__(cfg, stage, transform)

        self.images = self.masks = self._get_ade20k_pairs()
        if self.stage != 'test':
            assert len(self.images) == len(self.masks)

    
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.stage == 'test':
            if self.transform is not None:
                img, _ = self.transform(img, None)
            return img, None
        
        mask = img.Open(self.masks[index])

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
        mask = np.array(mask).astype('int64') - 1
        return torch.from_numpy(mask)

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 1

    


    def _get_ade20k_pairs(self):
        def get_path_pairs(img_foler, mask_folder):
            img_paths = []
            mask_paths = []
            for filename in os.listdir(img_foler):
                basename, _ = os.path.splitext(filename)
                if filename.endswith('.jpg'):
                    imgpath = os.path.join(img_foler, filename)
                    maskname = basename + '.png'
                    maskpath = os.path.join(mask_folder, maskname)
                    if os.path.isfile(maskpath) and os.path.isfile(imgpath):
                        img_paths.append(imgpath)
                        mask_paths.append(maskpath)
                    else:
                        print("Cannont find the image [{}] or mask [{}]",format(imgpath, maskpath))
            return img_paths, mask_paths

        if self.stage == 'train':
            img_folder = os.path.join(self.root, 'images/training')
            mask_folder = os.path.join(self.root, 'annotations/training')
            img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
            print('len(img_paths):', len(img_paths))
            assert len(img_paths) == 20210
        elif self.stage == 'val':
            img_folder = os.path.join(self.root, 'images/validation')
            mask_folder = os.path.join(self.root, 'annotations/validation')
            img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
            assert len(img_paths) == 2000
        elif self.stage == 'test':
            with open(os.path.join(self.root, 'annotations/testing.txt')) as f:
                img_paths = [os.path.join(self.root, 'images/testing', line.strip()) for line in f]
            assert len(img_paths) == 3352
            return img_paths, None
        else:
            assert self.stage == 'trainval'
            train_img_folder = os.path.join(folder, 'images/training')
            train_mask_folder = os.path.join(folder, 'annotations/training')
            val_img_folder = os.path.join(folder, 'images/validation')
            val_mask_folder = os.path.join(folder, 'annotations/validation')
            train_img_paths, train_mask_paths = get_path_pairs(train_img_folder, train_mask_folder)
            val_img_paths, val_mask_paths = get_path_pairs(val_img_folder, val_mask_folder)
            img_paths = train_img_paths + val_img_paths
            mask_paths = train_mask_paths + val_mask_paths
            assert len(img_paths) == 22210
        return img_paths, mask_paths