import torch
from torch import nn
from torch.nn import functional as F
from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class ESPCN(nn.Module):
    def __init__(self, in_channels=1, upscale_factor=4):
        super(ESPCN, self).__init__()
        self.feature_maps = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, stride=1, padding=5 // 2),
            nn.Tanh(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=3 // 2),
            nn.Tanh()
        )

        self.sub_pixel = nn.Sequential(
            nn.Conv2d(32, in_channels * upscale_factor ** 2, kernel_size=3, stride=1, padding=3 // 2),
            nn.PixelShuffle(upscale_factor)
        )

    def forward(self, x):
        out = self.feature_maps(x)
        out = self.sub_pixel(out)
        return out