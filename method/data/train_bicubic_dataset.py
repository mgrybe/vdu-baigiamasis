import cv2
import numpy as np
import random
import torch
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import img2tensor
from basicsr.utils.matlab_functions import imresize
from basicsr.utils.registry import DATASET_REGISTRY
from torch.utils import data as data

import h5py


@DATASET_REGISTRY.register()
class BicubicDataset(data.Dataset):
    """Dataset for training with bicubic degradation only.

    Loads GT images from an HDF5 file, applies augmentation, and generates
    LR images via MATLAB-style bicubic downsampling. Returns paired LQ/GT
    tensors directly (no on-GPU degradation pipeline).

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            h5_file (str): Path to the HDF5 file containing GT images.
            scale (int): Downsampling scale factor (e.g. 4).
            gt_size (int): GT crop size (LQ crop size = gt_size / scale).
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (vertical flip + transpose).
    """

    def __init__(self, opt):
        super(BicubicDataset, self).__init__()
        self.opt = opt
        self.h5_file = opt['h5_file']
        self.scale = opt['scale']

        with h5py.File(self.h5_file, 'r') as f:
            self.paths = sorted(list(f.keys()), key=lambda k: int(k) if k.isdigit() else k)

    def __getitem__(self, index):
        scale = self.scale
        gt_size = self.opt['gt_size']

        # -------------------------------- Load gt image -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.paths[index]
        with h5py.File(self.h5_file, 'r') as f:
            img_bytes = np.array(f[gt_path])  # HxW or HxWxC, uint8

        img_gt = img_bytes.astype(np.float32) / 255.

        # -------------------- modcrop to be divisible by scale -------------------- #
        h, w = img_gt.shape[0:2]
        h = h - h % scale
        w = w - w % scale
        img_gt = img_gt[0:h, 0:w, :]

        # -------------------- Ensure minimum size for cropping -------------------- #
        if h < gt_size or w < gt_size:
            h = max(h, gt_size)
            w = max(w, gt_size)
            img_gt = cv2.resize(img_gt, (w, h))

        # -------------------- Generate LR via bicubic downsampling -------------------- #
        img_lq = imresize(img_gt, 1 / scale)

        img_gt = np.ascontiguousarray(img_gt, dtype=np.float32)
        img_lq = np.ascontiguousarray(img_lq, dtype=np.float32)

        # -------------------- Paired random crop -------------------- #
        img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)

        # -------------------- Augmentation: flip, rotation -------------------- #
        # print(f'Augmentation: flip={self.opt["use_hflip"]}, rotation={self.opt["use_rot"]}')
        img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # -------------------- HWC to CHW, numpy to tensor -------------------- #
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=False, float32=True)

        return {'lq': img_lq, 'gt': img_gt, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)
