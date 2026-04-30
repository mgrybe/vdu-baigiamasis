import cv2
import numpy as np

from basicsr.metrics.metric_util import reorder_image, to_y_channel
from basicsr.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_laplacian_var(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate Laplacian Variance (sharpness/blurriness measure).

    A no-reference metric that measures the sharpness of an image by computing
    the variance of the Laplacian. Higher values indicate sharper images.

    Args:
        img (ndarray): SR output image with range [0, 255] (BGR, HWC by default).
        img2 (ndarray): Ground truth image (unused, accepted for signature compatibility).
        crop_border (int): Cropped pixels in each edge of an image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: Laplacian variance value. Higher means sharper.
    """
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        gray = np.squeeze(img).astype(np.float64)
    else:
        gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = np.var(laplacian)

    return float(variance)
