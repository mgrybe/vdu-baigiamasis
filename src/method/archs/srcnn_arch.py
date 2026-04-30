import torch
from torch import nn
from torch.nn import functional as F
from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class SRCNN(nn.Module):
    def __init__(self, num_in_ch=1, num_out_ch=1, num_feat=64, scale=4):
        super(SRCNN, self).__init__()
        self.scale = scale

        # Patch extraction and representation: features.0
        self.features = nn.Sequential(
            nn.Conv2d(num_in_ch, num_feat, kernel_size=9, padding=9//2),
            nn.ReLU(inplace=True)
        )
        # Non-linear mapping: map.0
        self.map = nn.Sequential(
            nn.Conv2d(num_feat, 32, kernel_size=5, padding=5//2),
            nn.ReLU(inplace=True)
        )
        # Reconstruction: reconstruction
        self.reconstruction = nn.Conv2d(32, num_out_ch, kernel_size=5, padding=5//2)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)

        x = self.features(x)
        x = self.map(x)
        x = self.reconstruction(x)
        return x