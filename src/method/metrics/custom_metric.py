import pyiqa
import cv2
import numpy as np
import torch
from basicsr.metrics.metric_util import reorder_image
from basicsr.utils.registry import METRIC_REGISTRY
import pyiqa

# ignore warnings
import warnings

warnings.filterwarnings("ignore")

def common_np2pyiqa(img1, img2, crop_border, input_order='HWC'):
    # todo: directly convert to tensor

    assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    # convert to pytorch tensor
    img1 = img1.transpose(2, 0, 1)
    img2 = img2.transpose(2, 0, 1)
    img1 = torch.from_numpy(img1).unsqueeze(0).float() / 255.0
    img2 = torch.from_numpy(img2).unsqueeze(0).float() / 255.0

    # BGR 2 RGB
    img1 = img1[:, [2, 1, 0], :, :]
    img2 = img2[:, [2, 1, 0], :, :]

    return img1, img2


lpips_metric = pyiqa.create_metric('lpips', device='cpu', as_loss=False)
dists_metric = pyiqa.create_metric('dists', device='cpu', as_loss=False)
niqe_metric = pyiqa.create_metric('niqe', device='cpu', as_loss=False)
maniqa_metric = pyiqa.create_metric('maniqa', device='cpu', as_loss=False)
musiq_metric = pyiqa.create_metric('musiq', device='cpu', as_loss=False)
clipiqa_metric = pyiqa.create_metric('clipiqa', device='cpu', as_loss=False)

@METRIC_REGISTRY.register()
def calculate_lpipsv3(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate lpips

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the LPIPS calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.

    Returns:
        float: lpips result.
    """
    with torch.no_grad():
        assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
        img1, img2 = common_np2pyiqa(img, img2, crop_border, input_order=input_order)

        lpips_score = lpips_metric(img1, img2).squeeze().item()
    return lpips_score

@METRIC_REGISTRY.register()
def calculate_distsv3(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate lpips

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the LPIPS calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.

    Returns:
        float: lpips result.
    """
    with torch.no_grad():
        assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
        img1, img2 = common_np2pyiqa(img, img2, crop_border, input_order=input_order)

        lpips_score = dists_metric(img1, img2).squeeze().item()
    return lpips_score

@METRIC_REGISTRY.register()
def calculate_niqev3(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate lpips

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the LPIPS calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.

    Returns:
        float: lpips result.
    """
    with torch.no_grad():
        assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
        img1, img2 = common_np2pyiqa(img, img2, crop_border, input_order=input_order)

        lpips_score = niqe_metric(img1, img2).squeeze().item()
    return lpips_score

@METRIC_REGISTRY.register()
def calculate_maniqav3(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate lpips

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the LPIPS calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.

    Returns:
        float: lpips result.
    """
    with torch.no_grad():
        assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
        img1, img2 = common_np2pyiqa(img, img2, crop_border, input_order=input_order)

        lpips_score = maniqa_metric(img1, img2).squeeze().item()
    return lpips_score

@METRIC_REGISTRY.register()
def calculate_musiqv3(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate lpips

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the LPIPS calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.

    Returns:
        float: lpips result.
    """
    with torch.no_grad():
        assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
        img1, img2 = common_np2pyiqa(img, img2, crop_border, input_order=input_order)

        lpips_score = musiq_metric(img1, img2).squeeze().item()
    return lpips_score

@METRIC_REGISTRY.register()
def calculate_clipiqav3(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate lpips

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the LPIPS calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.

    Returns:
        float: lpips result.
    """
    with torch.no_grad():
        assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
        img1, img2 = common_np2pyiqa(img, img2, crop_border, input_order=input_order)

        lpips_score = clipiqa_metric(img1, img2).squeeze().item()
    return lpips_score


