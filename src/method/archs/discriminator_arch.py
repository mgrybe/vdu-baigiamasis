from basicsr.utils.registry import ARCH_REGISTRY
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
import torch


@ARCH_REGISTRY.register()
class UNetDiscriminatorSN(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)

    It is used in Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    Arg:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(UNetDiscriminatorSN, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        # the first convolution
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convolutions
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out

@ARCH_REGISTRY.register()
class VGGStyleDiscriminator256(nn.Module):
    """VGG style discriminator with input size 256 x 256.

    It is now used to train VideoGAN.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features.
            Default: 64.
    """

    def __init__(self, num_in_ch, num_feat):
        super(VGGStyleDiscriminator256, self).__init__()

        self.conv0_0 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(num_feat, num_feat, 4, 2, 1, bias=False)
        self.bn0_1 = nn.SyncBatchNorm(num_feat, affine=True)

        self.conv1_0 = nn.Conv2d(num_feat, num_feat * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.SyncBatchNorm(num_feat * 2, affine=True)
        self.conv1_1 = nn.Conv2d(num_feat * 2, num_feat * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.SyncBatchNorm(num_feat * 2, affine=True)

        self.conv2_0 = nn.Conv2d(num_feat * 2, num_feat * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.SyncBatchNorm(num_feat * 4, affine=True)
        self.conv2_1 = nn.Conv2d(num_feat * 4, num_feat * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.SyncBatchNorm(num_feat * 4, affine=True)

        self.conv3_0 = nn.Conv2d(num_feat * 4, num_feat * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.SyncBatchNorm(num_feat * 8, affine=True)
        self.conv3_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.SyncBatchNorm(num_feat * 8, affine=True)

        self.conv4_0 = nn.Conv2d(num_feat * 8, num_feat * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.SyncBatchNorm(num_feat * 8, affine=True)
        self.conv4_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.SyncBatchNorm(num_feat * 8, affine=True)

        self.conv5_0 = nn.Conv2d(num_feat * 8, num_feat * 8, 3, 1, 1, bias=False)
        self.bn5_0 = nn.SyncBatchNorm(num_feat * 8, affine=True)
        self.conv5_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn5_1 = nn.SyncBatchNorm(num_feat * 8, affine=True)

        self.linear1 = nn.Linear(num_feat * 8 * 4 * 4, 100)
        self.linear2 = nn.Linear(100, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        assert x.size(2) == 256 and x.size(3) == 256, (f'Input spatial size must be 256x256, but received {x.size()}.')

        feat = self.lrelu(self.conv0_0(x))
        feat = self.lrelu(self.bn0_1(self.conv0_1(feat)))  # output spatial size: (128, 128)

        feat = self.lrelu(self.bn1_0(self.conv1_0(feat)))
        feat = self.lrelu(self.bn1_1(self.conv1_1(feat)))  # output spatial size: (64, 64)

        feat = self.lrelu(self.bn2_0(self.conv2_0(feat)))
        feat = self.lrelu(self.bn2_1(self.conv2_1(feat)))  # output spatial size: (32, 32)

        feat = self.lrelu(self.bn3_0(self.conv3_0(feat)))
        feat = self.lrelu(self.bn3_1(self.conv3_1(feat)))  # output spatial size: (16, 16)

        feat = self.lrelu(self.bn4_0(self.conv4_0(feat)))
        feat = self.lrelu(self.bn4_1(self.conv4_1(feat)))  # output spatial size: (8, 8)

        feat = self.lrelu(self.bn5_0(self.conv5_0(feat)))
        feat = self.lrelu(self.bn5_1(self.conv5_1(feat)))  # output spatial size: (4, 4)

        feat = feat.view(feat.size(0), -1)
        feat = self.lrelu(self.linear1(feat))
        out = self.linear2(feat)
        return out

@ARCH_REGISTRY.register()
class VGGStyleDiscriminator128(nn.Module):
    """VGG style discriminator with input size 128 x 128."""

    def __init__(self, num_in_ch, num_feat):
        super(VGGStyleDiscriminator128, self).__init__()

        self.conv0_0 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(num_feat, num_feat, 4, 2, 1, bias=False)
        self.bn0_1 = nn.SyncBatchNorm(num_feat, affine=True)

        self.conv1_0 = nn.Conv2d(num_feat, num_feat * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.SyncBatchNorm(num_feat * 2, affine=True)
        self.conv1_1 = nn.Conv2d(num_feat * 2, num_feat * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.SyncBatchNorm(num_feat * 2, affine=True)

        self.conv2_0 = nn.Conv2d(num_feat * 2, num_feat * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.SyncBatchNorm(num_feat * 4, affine=True)
        self.conv2_1 = nn.Conv2d(num_feat * 4, num_feat * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.SyncBatchNorm(num_feat * 4, affine=True)

        self.conv3_0 = nn.Conv2d(num_feat * 4, num_feat * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.SyncBatchNorm(num_feat * 8, affine=True)
        self.conv3_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.SyncBatchNorm(num_feat * 8, affine=True)

        self.conv4_0 = nn.Conv2d(num_feat * 8, num_feat * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.SyncBatchNorm(num_feat * 8, affine=True)
        self.conv4_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.SyncBatchNorm(num_feat * 8, affine=True)

        self.linear1 = nn.Linear(num_feat * 8 * 4 * 4, 100)
        self.linear2 = nn.Linear(100, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        assert x.size(2) == 128 and x.size(3) == 128, (f'Input spatial size must be 128x128, but received {x.size()}.')

        feat = self.lrelu(self.conv0_0(x))
        feat = self.lrelu(self.bn0_1(self.conv0_1(feat)))  # output spatial size: (64, 64)

        feat = self.lrelu(self.bn1_0(self.conv1_0(feat)))
        feat = self.lrelu(self.bn1_1(self.conv1_1(feat)))  # output spatial size: (32, 32)

        feat = self.lrelu(self.bn2_0(self.conv2_0(feat)))
        feat = self.lrelu(self.bn2_1(self.conv2_1(feat)))  # output spatial size: (16, 16)

        feat = self.lrelu(self.bn3_0(self.conv3_0(feat)))
        feat = self.lrelu(self.bn3_1(self.conv3_1(feat)))  # output spatial size: (8, 8)

        feat = self.lrelu(self.bn4_0(self.conv4_0(feat)))
        feat = self.lrelu(self.bn4_1(self.conv4_1(feat)))  # output spatial size: (4, 4)

        feat = feat.view(feat.size(0), -1)
        feat = self.lrelu(self.linear1(feat))
        out = self.linear2(feat)
        return out

@ARCH_REGISTRY.register()
class MOD(nn.Module):
    """
    CAL-GAN Mixture-of-Classifiers (MoC) Discriminator.

    Pipeline (Figure 2, Sections 3.1-3.4):
        1. Feature Extractor (FE):  I -> F ∈ R^{B × c × H/4 × W/4}
        2. Router (Eq. 2):          F_HR -> R via Gumbel-Softmax (HR only)
        3. Spatial masking (Eq. 3): F_i = F ⊙ R_i  (broadcast across channels)
        4. Orthogonal conv (Eq. 5): F*_i = O_i(F_i)  (N independent 1×1 convs)
        5. Classifiers (Eq. 8):     D = Σ C_i(F*_i)  (N independent classifiers)

    Usage with BasicSR training loop:
        # HR call — computes routing internally
        real_pred, routing, features, ortho_w = net_d(hr_img)
        # SR call — reuses HR routing
        fake_pred, _, _, _ = net_d(sr_img, routing.detach())
    """

    def __init__(self, num_in_ch=3, num_feat=128, num_expert=12):
        super(MOD, self).__init__()
        self.num_expert = num_expert
        self.num_feat = num_feat
        c_out = num_feat * 4  # c = 512 in paper (Section 3.2)

        # ── Feature Extractor (FE) ──────────────────────────────────────
        # Conv-BN-LeakyReLU backbone. Two stride-2 convs → H/4 × W/4.
        self.FE = nn.Sequential(
            nn.Conv2d(num_in_ch, num_feat, 3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_feat, num_feat, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_feat),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_feat, num_feat * 2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_feat * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_feat * 2, num_feat * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_feat * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_feat * 2, c_out, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU(0.2, True),
        )

        # ── Router (Eq. 2) ──────────────────────────────────────────────
        # 1×1 conv producing N routing logits per spatial location.
        self.router = nn.Conv2d(c_out, num_expert, kernel_size=1)

        # ── Orthogonal Convolutions O_i (Eq. 5) ────────────────────────
        # N independent 1×1 convolutions (no bias).
        self.ortho_convs = nn.ModuleList([
            nn.Conv2d(c_out, c_out, kernel_size=1, bias=False)
            for _ in range(num_expert)
        ])

        # ── Mixture of Classifiers C_i (Eq. 8) ─────────────────────────
        # N independent expert classifiers.
        self.classifiers = nn.ModuleList([
            self._build_classifier(c_out, num_feat)
            for _ in range(num_expert)
        ])

    @staticmethod
    def _build_classifier(in_ch, num_feat):
        """Single expert classifier using 1×1 convs (per-pixel Linear)."""
        return nn.Sequential(
            nn.Conv2d(in_ch, num_feat // 2, kernel_size=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_feat // 2, 1, kernel_size=1),
        )

    def forward(self, x, routing=None, tau=1.0, hard=True):
        """
        Single-image forward pass, compatible with BasicSR training loop.

        The routing mask is computed only on the first call (HR image) and
        reused in subsequent calls (SR image) by passing it back in.
        This implements: "routing mask R is only obtained from I_HR" (Fig. 2).

        Args:
            x:       Input image (B, 3, H, W) — either HR or SR.
            routing: Pre-computed hard routing mask R from the HR call,
                     shape (B, N, H/4, W/4). If None, routing is computed
                     from x (this should be the HR path).
            tau:     Gumbel-Softmax temperature (annealed during training).
            hard:    If True, one-hot forward with straight-through grads.

        Returns:
            output:        Discriminator prediction (B, 1, H/4, W/4).
            routing_out:   Hard one-hot mask R (B, N, H/4, W/4) — pass
                           (detached) to SR call to share the same routing.
            features:      List of N features F*_i, each (B, c, H/4, W/4),
                           for computing LDA loss (Eq. 7) externally.
            ortho_weights: Stacked ortho conv weights (N, c, c, 1, 1),
                           for computing orthogonal loss (Eq. 6) externally.

        Note:
            On HR calls (routing=None), the soft routing distribution R'
            needed for the balancing loss (Eq. 4) is stored in self._R_prime.
            Access it as: net_d._R_prime after the HR forward call.
        """
        # ── 1. Extract features ─────────────────────────────────────────
        F_feat = self.FE(x)  # (B, c, H/4, W/4)

        # ── 2. Routing (Eq. 2) ──────────────────────────────────────────
        if routing is None:
            # HR path: compute routing from features
            routing_logits = self.router(F_feat)  # (B, N, H/4, W/4)

            # R': standard softmax — stored for balancing loss (Eq. 4)
            self._R_prime = F.softmax(routing_logits, dim=1)

            # R: Gumbel-Softmax one-hot mask (Eq. 2)
            R = F.gumbel_softmax(routing_logits, tau=tau, hard=hard, dim=1)
        else:
            # SR path: reuse the hard mask from HR call
            R = routing

        # ── 3-5. Spatial masking → orthogonal projection → classify ─────
        expert_outputs = []
        F_stars = []

        for i in range(self.num_expert):
            # Eq. 3: spatial masking — R_i broadcast across channel dim
            R_i = R[:, i:i+1, :, :]             # (B, 1, H/4, W/4)
            F_i = F_feat * R_i                   # (B, c, H/4, W/4)

            # Eq. 5: orthogonal 1×1 convolution
            F_i_star = self.ortho_convs[i](F_i)  # (B, c, H/4, W/4)
            F_stars.append(F_i_star)

            # Eq. 8: expert classifier, masked by R_i to enforce spatial
            # disjointness. Without this, classifier bias terms produce
            # non-zero output at zeroed-out locations, contaminating the sum.
            C_i = self.classifiers[i](F_i_star) * R_i  # (B, 1, H/4, W/4)
            expert_outputs.append(C_i)

        # Eq. 8: D = Σ C_i(F*_i) — each pixel has exactly one non-zero term
        output = sum(expert_outputs)

        # Gather orthogonal weights for L_o (Eq. 6)
        ortho_weights = torch.stack([conv.weight for conv in self.ortho_convs])

        return output, R, F_stars, ortho_weights # output: (B, 1, H/4, W/4), R: (B, N, H/4, W/4), F_stars: (N, B, c, H/4, W/4), ortho_weights: (N, c, c, 1, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# MODv2: Authors' actual CAL-GAN architecture
#
# Key differences from MOD (paper equations):
#   - No Gumbel-Softmax: plain softmax + hard argmax
#   - No spatial masking: CodeReduction channel-splits features
#   - CodeReduction replaces N independent ortho convs
#   - Independent classifiers on channel subsets
# ═══════════════════════════════════════════════════════════════════════════════

class OrthorTransform(nn.Module):
    """Element-wise learnable weight for orthogonal regularization."""
    def __init__(self, c_dim, feat_hw, groups):
        super(OrthorTransform, self).__init__()
        self.groups = groups
        self.weight = nn.Parameter(torch.randn(1, feat_hw, c_dim))

    def forward(self, feat):
        pred = feat * self.weight.expand_as(feat)
        return pred, self.weight.view(self.groups, -1)


class CodeReduction(nn.Module):
    """Expand features to c*N channels, apply ortho transform, split into N chunks."""
    def __init__(self, c_dim, feat_hw, blocks=4):
        super(CodeReduction, self).__init__()
        self.body = nn.Sequential(
            nn.Linear(c_dim, c_dim * blocks),
            nn.LeakyReLU(0.2, True)
        )
        self.trans = OrthorTransform(c_dim=c_dim * blocks, feat_hw=feat_hw, groups=blocks)
        self.leakyrelu = nn.LeakyReLU(0.2, True)

    def forward(self, feat):
        feat = self.body(feat)
        feat, weight = self.trans(feat)
        feat = self.leakyrelu(feat)
        return feat, weight


@ARCH_REGISTRY.register()
class MODv2(nn.Module):
    """
    CAL-GAN discriminator — authors' actual implementation.

    Unlike MOD (which implements the paper's equations), MODv2 uses:
      - Channel splitting via CodeReduction (no spatial masking)
      - Plain softmax + hard argmax routing (no Gumbel-Softmax)
      - Router only selects which expert OUTPUT to use per pixel
      - All experts see all spatial locations (different channel projections)

    This architecture specializes routing ~100x faster than MOD because
    CodeReduction handles feature separation without the chicken-and-egg
    problem inherent in spatial masking.
    """

    def __init__(self, num_in_ch=3, num_feat=64, num_expert=12):
        super(MODv2, self).__init__()
        self.num_expert = num_expert
        self.num_feat = num_feat
        c_out = num_feat * 4  # 256 channels (authors' original)

        # ── Feature Extractor (FE) ──────────────────────────────────────
        self.FE = nn.Sequential(
            nn.Conv2d(num_in_ch, num_feat, 3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_feat, num_feat, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_feat),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_feat, num_feat * 2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_feat * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_feat * 2, num_feat * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_feat * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_feat * 2, c_out, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU(0.2, True),
        )

        # ── Router: learnable weight matrix ─────────────────────────────
        self.w_gating = nn.Parameter(torch.randn(c_out, num_expert))

        # ── CodeReduction: shared Linear + ortho transform ──────────────
        self.orthonet = CodeReduction(
            c_dim=c_out, feat_hw=1, blocks=num_expert
        )

        # ── N independent classifiers ───────────────────────────────────
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(c_out, num_feat // 2),
                nn.LeakyReLU(0.2, True),
                nn.Linear(num_feat // 2, 1)
            )
            for _ in range(num_expert)
        ])

    def forward(self, x, routing=None, tau=1.0, hard=True):
        """
        Forward pass. tau and hard are accepted for API compatibility
        with the training loop but are ignored (MODv2 uses clean argmax).

        Returns:
            output:       (B, 1, H/4, W/4) discriminator prediction.
            routing:      (B, N, H/4, W/4) soft routing probabilities.
            features:     List of N tensors, each (B, HW/16, c_out),
                          channel-split CodeReduction output.
            ortho_weight: (N, c_out) ortho transform weight for L_o.
        """
        feat = self.FE(x)
        B, C, H, W = feat.shape
        feat_flat = feat.view(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)

        if routing is None:
            routing_flat = torch.einsum('bnd,de->bne', feat_flat, self.w_gating)
            routing_flat = routing_flat.softmax(dim=-1)  # (B, HW, N)
        else:
            # Reshape spatial routing back to flat form
            routing_flat = routing.permute(0, 2, 3, 1).reshape(B, H * W, self.num_expert)

        # Store soft routing for balancing loss (NO detach — gradient must flow to router)
        self._R_prime = routing_flat.permute(0, 2, 1).reshape(B, self.num_expert, H, W)

        # CodeReduction: expand + ortho transform + split
        feat_transformed, ortho_weight = self.orthonet(feat_flat)
        features = torch.split(
            feat_transformed,
            [feat_transformed.shape[-1] // self.num_expert] * self.num_expert,
            dim=-1
        )

        # Hard routing via argmax — no gradient through this path
        routing_top = torch.max(routing_flat, dim=-1)[1].unsqueeze(-1).float()
        for i in range(self.num_expert):
            if i == 0:
                output = self.classifiers[0](features[0])
            else:
                output = torch.where(
                    routing_top == i,
                    self.classifiers[i](features[i]),
                    output
                )

        # Reshape to spatial format for GAN loss compatibility
        # output is (B, HW, 1) from classifiers → (B, 1, H, W)
        output = output.squeeze(-1).view(B, H, W).unsqueeze(1)

        # Routing in spatial format for reuse in fake forward pass
        routing_spatial = routing_flat.permute(0, 2, 1).reshape(B, self.num_expert, H, W)

        return output, routing_spatial, list(features), ortho_weight
