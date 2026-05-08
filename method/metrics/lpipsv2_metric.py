import cv2
import numpy as np
import torch
import torch.nn.functional as F

from basicsr.metrics.metric_util import reorder_image, to_y_channel
from basicsr.utils.color_util import rgb2ycbcr_pt
from basicsr.utils.registry import METRIC_REGISTRY

import pyiqa
import lpips

# https://github.com/csjliang/LDL/blob/2d22dddc8e427acb7b112e7ee0a1c370edc3a5b9/scripts/metrics/table_calculate_lpips_all.py#L42
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
iqa_metric = lpips.LPIPS(net='alex').cuda() if torch.cuda.is_available() else lpips.LPIPS(net='alex').cpu()

def img2tensor(img):
    """Convert numpy image to tensor: HWC -> NCHW, [0, 255] -> [0, 1]"""
    img = img.astype(np.float32)
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0) / 255.0
    img = img[:, [2, 1, 0], :, :] # BGR 2 RGB
    img = 2 * img  - 1 # normalizing to [-1, 1]

    return img.to(device)

@METRIC_REGISTRY.register()
def calculate_lpipsv2(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):

    # (680, 1024, 3)
    # (680, 1024, 3)
    # HWC

    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
        img2 = img2.transpose(1, 2, 0)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)
        # to_y_channel returns (H, W, 1). For LPIPS we usually need 3 channels.
        img = np.concatenate([img] * 3, axis=2)
        img2 = np.concatenate([img2] * 3, axis=2)

    # Convert to tensors
    img_tensor = img2tensor(img)
    img2_tensor = img2tensor(img2)

    with torch.no_grad():
        result = iqa_metric(img_tensor, img2_tensor)

    return result.item()