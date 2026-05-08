import numpy as np
import torch

from basicsr.metrics.metric_util import reorder_image, to_y_channel
from basicsr.utils.registry import METRIC_REGISTRY

import pyiqa

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

iqa_metric = pyiqa.create_metric('musiq', device=device)


def img2tensor(img):
    """Convert numpy image to tensor: HWC -> NCHW, [0, 255] -> [0, 1]"""
    img = img.astype(np.float32) / 255.
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    img = img[:, [2, 1, 0], :, :]  # BGR 2 RGB
    return img.to(device)


@METRIC_REGISTRY.register()
def calculate_musiq(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate MUSIQ (Multi-Scale Image Quality Transformer).

    A no-reference metric. Higher values indicate better quality.

    Args:
        img (ndarray): SR output image with range [0, 255] (BGR, HWC by default).
        img2 (ndarray): Ground truth image (unused, accepted for signature compatibility).
        crop_border (int): Cropped pixels in each edge of an image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: MUSIQ score. Higher means better quality.
    """
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img = np.concatenate([img] * 3, axis=2)

    img_tensor = img2tensor(img)

    with torch.no_grad():
        result = iqa_metric(img_tensor)

    return result.item()
