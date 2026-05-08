# flake8: noqa
import os.path as osp

import sys
from types import ModuleType

# Monkey patch to fix compatibility between BasicSR and newer torchvision
try:
    import torchvision.transforms.functional as F
    # Create a dummy module to satisfy the old import path
    mock_module = ModuleType("torchvision.transforms.functional_tensor")
    mock_module.rgb_to_grayscale = F.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = mock_module
except ImportError:
    pass

# Fix for basicsr ImportError: rgb2ycbcr moved from matlab_functions to color_util
try:
    import basicsr.utils.matlab_functions as matlab_utils
    if not hasattr(matlab_utils, 'rgb2ycbcr'):
        try:
            from basicsr.utils import color_util
            matlab_utils.rgb2ycbcr = color_util.rgb2ycbcr
        except ImportError:
            pass
except ImportError:
    pass

import method.archs
import method.data
import method.models
import method.metrics
import method.losses
from basicsr.test import test_pipeline

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
