import cv2
import io
import math
import numpy as np
import random

from PIL import Image
from torch.utils import data as data

from basicsr.data.degradations import (
    circular_lowpass_kernel,
    random_mixed_kernels,
    random_add_gaussian_noise,
    random_add_poisson_noise,
)
from basicsr.utils import img2tensor
from basicsr.utils.img_process_util import usm_sharp
from basicsr.utils.matlab_functions import imresize
from basicsr.utils.registry import DATASET_REGISTRY

import h5py


INTERPOLATION_MODES = {
    'area': cv2.INTER_AREA,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
}


def _jpeg_compress(img, quality):
    """JPEG compression using PIL with 4:4:4 subsampling to match DiffJPEG.

    DiffJPEG (used in training) does NOT do chroma subsampling. OpenCV JPEG
    does 4:2:0 by default, producing different color artifacts the model wasn't
    trained on. PIL with subsampling=0 gives 4:4:4 (no subsampling).

    Args:
        img: float32 RGB image, HWC, range [0, 1].
        quality: JPEG quality (0-100).
    Returns:
        float32 RGB image, HWC, range [0, 1].
    """
    img = np.clip(img, 0, 1)
    img_uint8 = (img * 255.0).round().astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG', quality=int(quality), subsampling=0)
    buffer.seek(0)
    compressed = np.array(Image.open(buffer)).astype(np.float32) / 255.0
    return compressed


@DATASET_REGISTRY.register()
class EvalRealWorldPairedDataset(data.Dataset):
    """Evaluation dataset with Real-ESRGAN two-stage degradation pipeline.

    Applies the same degradation used during Real-world SR training (blur, resize,
    noise, JPEG compression — applied twice) but using CPU-compatible numpy/OpenCV
    operations with deterministic seeding for reproducible evaluation.
    """

    def __init__(self, opt):
        super(EvalRealWorldPairedDataset, self).__init__()
        self.opt = opt
        self.h5_file = opt['h5_file']
        self.dataset = opt['dataset']
        self.manual_seed = opt.get('manual_seed', 0)

        with h5py.File(self.h5_file, 'r') as f:
            group = f[self.dataset] if self.dataset is not None else f
            self.paths = sorted(list(group.keys()), key=lambda k: int(k) if k.isdigit() else k)

        # First degradation blur settings
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']
        self.blur_sigma = opt['blur_sigma']
        self.betag_range = opt['betag_range']
        self.betap_range = opt['betap_range']
        self.sinc_prob = opt['sinc_prob']

        # Second degradation blur settings
        self.blur_kernel_size2 = opt['blur_kernel_size2']
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']
        self.sinc_prob2 = opt['sinc_prob2']

        self.final_sinc_prob = opt['final_sinc_prob']

        # Kernel size ranges from 7 to 21 (matching train_dataset.py)
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]

    def _generate_kernel(self, kernel_list, kernel_prob, blur_sigma, betag_range, betap_range, sinc_prob):
        """Generate a blur kernel (matches train_dataset.py logic)."""
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < sinc_prob:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                kernel_list,
                kernel_prob,
                kernel_size,
                blur_sigma,
                blur_sigma,
                [-math.pi, math.pi],
                betag_range,
                betap_range,
                noise_range=None,
            )
        # Pad kernel to 21x21
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
        return kernel

    def _generate_sinc_kernel(self):
        """Generate the final sinc kernel (matches train_dataset.py logic)."""
        if np.random.uniform() < self.final_sinc_prob:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
        else:
            sinc_kernel = np.zeros((21, 21), dtype=np.float32)
            sinc_kernel[10, 10] = 1.0
        return sinc_kernel

    def _apply_degradation(self, img, ori_h, ori_w):
        """Apply two-stage Real-ESRGAN degradation (mirrors realhatgan_model.py:feed_data).

        All operations are either channel-agnostic (blur, resize, noise) or use
        PIL for JPEG (native RGB, 4:4:4 subsampling matching DiffJPEG). No
        BGR conversion needed.
        """
        scale = self.opt['scale']

        # Generate kernels
        kernel1 = self._generate_kernel(
            self.kernel_list, self.kernel_prob, self.blur_sigma,
            self.betag_range, self.betap_range, self.sinc_prob,
        )
        kernel2 = self._generate_kernel(
            self.kernel_list2, self.kernel_prob2, self.blur_sigma2,
            self.betag_range2, self.betap_range2, self.sinc_prob2,
        )
        sinc_kernel = self._generate_sinc_kernel()

        # ---- First degradation ---- #
        # Blur
        out = cv2.filter2D(img, -1, kernel1, borderType=cv2.BORDER_REFLECT_101)

        # Random resize
        updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
        if updown_type == 'up':
            resize_scale = np.random.uniform(1, self.opt['resize_range'][1])
        elif updown_type == 'down':
            resize_scale = np.random.uniform(self.opt['resize_range'][0], 1)
        else:
            resize_scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        h, w = out.shape[:2]
        new_h, new_w = int(h * resize_scale), int(w * resize_scale)
        if new_h > 0 and new_w > 0:
            out = cv2.resize(out, (new_w, new_h), interpolation=INTERPOLATION_MODES[mode])

        # Noise
        if np.random.uniform() < self.opt['gaussian_noise_prob']:
            out = random_add_gaussian_noise(
                out, sigma_range=self.opt['noise_range'], clip=True,
                rounds=False, gray_prob=self.opt['gray_noise_prob'],
            )
        else:
            out = random_add_poisson_noise(
                out, scale_range=self.opt['poisson_scale_range'],
                gray_prob=self.opt['gray_noise_prob'], clip=True, rounds=False,
            )

        # JPEG compression (PIL with 4:4:4 subsampling matching DiffJPEG)
        jpeg_quality = int(np.random.uniform(*self.opt['jpeg_range']))
        out = _jpeg_compress(out, quality=jpeg_quality)

        # ---- Second degradation ---- #
        # Blur
        if np.random.uniform() < self.opt['second_blur_prob']:
            out = cv2.filter2D(out, -1, kernel2, borderType=cv2.BORDER_REFLECT_101)

        # Random resize
        updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
        if updown_type == 'up':
            resize_scale = np.random.uniform(1, self.opt['resize_range2'][1])
        elif updown_type == 'down':
            resize_scale = np.random.uniform(self.opt['resize_range2'][0], 1)
        else:
            resize_scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        target_h = int(ori_h / scale * resize_scale)
        target_w = int(ori_w / scale * resize_scale)
        if target_h > 0 and target_w > 0:
            out = cv2.resize(out, (target_w, target_h), interpolation=INTERPOLATION_MODES[mode])

        # Noise
        if np.random.uniform() < self.opt['gaussian_noise_prob2']:
            out = random_add_gaussian_noise(
                out, sigma_range=self.opt['noise_range2'], clip=True,
                rounds=False, gray_prob=self.opt['gray_noise_prob2'],
            )
        else:
            out = random_add_poisson_noise(
                out, scale_range=self.opt['poisson_scale_range2'],
                gray_prob=self.opt['gray_noise_prob2'], clip=True, rounds=False,
            )

        # JPEG compression + final sinc filter (two orderings, 50% each)
        final_h = ori_h // scale
        final_w = ori_w // scale
        if np.random.uniform() < 0.5:
            # Resize back + sinc filter, then JPEG
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = cv2.resize(out, (final_w, final_h), interpolation=INTERPOLATION_MODES[mode])
            out = cv2.filter2D(out, -1, sinc_kernel, borderType=cv2.BORDER_REFLECT_101)
            jpeg_quality = int(np.random.uniform(*self.opt['jpeg_range2']))
            out = _jpeg_compress(out, quality=jpeg_quality)
        else:
            # JPEG, then resize back + sinc filter
            jpeg_quality = int(np.random.uniform(*self.opt['jpeg_range2']))
            out = _jpeg_compress(out, quality=jpeg_quality)
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = cv2.resize(out, (final_w, final_h), interpolation=INTERPOLATION_MODES[mode])
            out = cv2.filter2D(out, -1, sinc_kernel, borderType=cv2.BORDER_REFLECT_101)

        # Clamp and round
        out = np.clip((out * 255.0).round(), 0, 255) / 255.0

        return out.astype(np.float32)

    def __getitem__(self, index):
        scale = self.opt['scale']

        # Load GT image from HDF5
        gt_path = self.paths[index]
        with h5py.File(self.h5_file, 'r') as f:
            group = f[self.dataset] if self.dataset is not None else f
            img_bytes = np.array(group[self.paths[index]])
        img_gt = img_bytes.astype(np.float32) / 255.0

        # Modcrop
        size_h, size_w, _ = img_gt.shape
        size_h = size_h - size_h % scale
        size_w = size_w - size_w % scale
        img_gt = img_gt[0:size_h, 0:size_w, :]

        # Ensure minimum size
        size_h = max(size_h, self.opt['gt_size'])
        size_w = max(size_w, self.opt['gt_size'])
        img_gt = cv2.resize(img_gt, (size_w, size_h))

        img_gt = np.ascontiguousarray(img_gt, dtype=np.float32)

        # Seed RNG deterministically for this image
        seed = self.manual_seed * 10000 + index
        old_np_state = np.random.get_state()
        old_py_state = random.getstate()
        np.random.seed(seed)
        random.seed(seed)

        try:
            # USM sharpen GT for degradation input (matches training: gt_usm feeds degradation)
            use_usm = self.opt.get('use_usm', True)
            img_for_degrade = usm_sharp(img_gt.copy()) if use_usm else img_gt.copy()

            # Apply two-stage real-world degradation
            img_lq = self._apply_degradation(img_for_degrade, size_h, size_w)
        finally:
            np.random.set_state(old_np_state)
            random.setstate(old_py_state)

        # Crop GT to match LQ * scale
        img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=False, float32=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': ((self.dataset + '_') if self.dataset is not None else '') + f'{index}_' + gt_path,
            'gt_path': gt_path,
        }

    def __len__(self):
        return len(self.paths)
