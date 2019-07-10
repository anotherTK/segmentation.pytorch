
import os
import argparse
from PIL import Image
import numpy as np
import cv2

Image.MAX_IMAGE_PIXELS = 100000000000


def main(args):

    # identify test file
    origin_files = os.listdir(args.test)
    origin_files_id = [e.split('.')[0] for e in origin_files]

    for _id in origin_files_id:
        origin_img = Image.open(os.path.join(args.test, _id + '.png'))
        width, height = origin_img.size
        result_label = np.zeros((height, width), dtype=np.uint8)
        save_path = os.path.join(args.save, _id + '_predict.png')

        # identify test splited files
        splited_files = [e for e in os.listdir(args.test_s) if _id in e]
        
        # identify test segmented files
        segmented_files = [e for e in os.listdir(args.test_r) if _id in e]

        assert len(splited_files) == len(segmented_files)

        for filename in splited_files:
            splited_img = Image.open(os.path.join(args.test_s, filename))
            s_w, s_h = splited_img.size
            segmed_img = Image.open(os.path.join(args.test_r, filename))
            segmed_img = segmed_img.resize((s_w, s_h), Image.NEAREST)
            segmed_img = np.array(segmed_img)
            _name = filename.split('.')[0]
            c_idx = int(_name.split('_')[-1])
            r_idx = int(_name.split('_')[-2])
            x = c_idx * args.stride
            y = r_idx * args.stride
            result_label[y:y + args.crop_size, x:x + args.crop_size] = segmed_img

        cv2.imwrite(save_path, result_label)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit style results")
    parser.add_argument('--test', default='./work_dirs/remote_sensing/test_a', help='test origin file')
    parser.add_argument('--test-s', default='./datasets/rsv/images/test', help='test splited file')
    parser.add_argument('--test-r', default='./work_dirs/rsv/inference/rsv', help='test detected results')
    parser.add_argument('--save', default='./work_dirs/remote_sensing/predicted')
    parser.add_argument('--crop_size', type=int, default=520)
    parser.add_argument('--stride', type=int, default=260)

    args = parser.parse_args()
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    main(args)
