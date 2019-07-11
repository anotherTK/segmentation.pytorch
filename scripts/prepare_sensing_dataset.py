
import os
import cv2
from PIL import Image
import argparse
import numpy as np
import random
import shutil

Image.MAX_IMAGE_PIXELS = 100000000000

def p_train_val(args):
    crop_size = int(args.crop_size)
    stride = int(args.stride)

    train_image_savepath = os.path.join(args.save, 'images', 'train')
    val_image_savepath = os.path.join(args.save, 'images', 'val')
    train_label_savepath = os.path.join(args.save, 'annotations', 'train')
    val_label_savepath = os.path.join(args.save, 'annotations', 'val')

    if not os.path.exists(train_image_savepath):
        os.makedirs(train_image_savepath)
        os.makedirs(val_image_savepath)
        os.makedirs(train_label_savepath)
        os.makedirs(val_label_savepath)

    files = os.listdir(os.path.join(args.root, 'train'))
    images = [e for e in files if 'label' not in e]
    labels = [e for e in files if 'label' in e]
    images = sorted(images)
    labels = sorted(labels)
    assert len(images) == len(labels)

    # [image name]_[row index]_[col index]
    for i, image in enumerate(images):
        img = Image.open(os.path.join(args.root, 'train', image))
        # TODO: Does the alpha channel helps?
        img = img.convert("RGB")
        lbl = Image.open(os.path.join(args.root, 'train', labels[i]))
        
        img = np.array(img)
        lbl = np.array(lbl)

        height, width, _ = img.shape

        # split into small image patchs
        r_num = (height - crop_size) // stride + 1
        c_num = (width - crop_size) // stride + 1
        for r_idx in range(r_num):
            for c_idx in range(c_num):
                x = c_idx * stride
                y = r_idx * stride
                crop_img = img[y:y + crop_size, x:x + crop_size]
                gray_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
                if gray_img.max() < 30:
                    continue
                crop_lbl = lbl[y:y + crop_size, x:x + crop_size]
                image_name = image.split('.')[0] + "_{}_{}.png".format(r_idx, c_idx)
                label_name = image_name.split('.')[0] + "_label.png"

                cv2.imwrite(os.path.join(train_image_savepath, image_name), crop_img)
                cv2.imwrite(os.path.join(train_label_savepath, label_name), crop_lbl)

    total_samples = len(os.listdir(train_image_savepath))
    print("There are {} valid images for training and validation".format(total_samples))

    # choose the validation set
    # TODO bugs: if stride < crop_size, validation set would share some areas with train set. ---> only used for validating the algorithm with stride=crop_size case
    split_ratio = float(args.ratio)

    val_num = int(total_samples * split_ratio)
    stride_num = 1 if stride >= crop_size else ((crop_size - 1) // stride + 1)
    valid_files = []
    for file in os.listdir(train_image_savepath):
        c_idx = int(file.split('.')[0].split('_')[-1])
        r_idx = int(file.split('.')[0].split('_')[-2])
        if (c_idx % stride_num == 0) and (r_idx % stride_num == 0):
            valid_files.append(file)

    random.shuffle(valid_files)

    # move the selected files
    for i in range(val_num):
        shutil.move(
            os.path.join(train_image_savepath, valid_files[i]), 
            os.path.join(val_image_savepath, valid_files[i])
        )
        shutil.move(
            os.path.join(train_label_savepath, valid_files[i].split('.')[0] + "_label.png"),
            os.path.join(val_label_savepath,valid_files[i].split('.')[0] + "_label.png")
        )


    print("There are {} images for training".format(len(os.listdir(train_image_savepath))))
    print("There are {} images for validation".format(len(os.listdir(val_image_savepath))))

def p_test(args):
    crop_size = int(args.crop_size)
    stride = int(args.stride)

    test_image_savepath = os.path.join(args.save, 'images', 'test')

    if not os.path.exists(test_image_savepath):
        os.makedirs(test_image_savepath)

    images = os.listdir(os.path.join(args.root, 'test_a'))
    images = sorted(images)

    # [image name]_[row index]_[col index]
    for image in images:
        img = Image.open(os.path.join(args.root, 'test_a', image))
        # TODO: Does the alpha channel helps?
        img = img.convert("RGB")
        img = np.array(img)
        height, width, _ = img.shape

        # split into small image patchs
        r_num = (height - crop_size) // stride + 1
        c_num = (width - crop_size) // stride + 1
        for r_idx in range(r_num):
            for c_idx in range(c_num):
                x = c_idx * stride
                y = r_idx * stride
                crop_img = img[y:y + crop_size, x:x + crop_size]
                gray_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
                if gray_img.max() < 30:
                    continue

                image_name = image.split('.')[0] + "_{}_{}.png".format(r_idx, c_idx)

                cv2.imwrite(os.path.join(
                    test_image_savepath, image_name), crop_img)

    total_samples = len(os.listdir(test_image_savepath))
    print("There are {} valid images for test".format(total_samples))


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Split into small images")
    parser.add_argument('--root', default='work_dirs/remote_sensing/')
    parser.add_argument('--save', default='datasets/rsv')
    parser.add_argument('--crop-size', type=int ,default=520)
    parser.add_argument('--stride', type=int, default=260)
    parser.add_argument('--ratio', type=float, default=0.02)

    args = parser.parse_args()

    p_train_val(args)
    p_test(args)
