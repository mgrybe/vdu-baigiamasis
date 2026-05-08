import cv2
import numpy as np
import torch
import torch.nn.functional as F

from basicsr.metrics.metric_util import reorder_image, to_y_channel
from basicsr.utils.color_util import rgb2ycbcr_pt
from basicsr.utils.registry import METRIC_REGISTRY

import pyiqa

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# create metric with default setting
iqa_metric = pyiqa.create_metric('dists', device=device)

def img2tensor(img):
    """Convert numpy image to tensor: HWC -> NCHW, [0, 255] -> [0, 1]"""
    img = img.astype(np.float32) / 255.
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    img = img[:, [2, 1, 0], :, :]  # BGR 2 RGB
    return img.to(device)

@METRIC_REGISTRY.register()
def calculate_dists(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):

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