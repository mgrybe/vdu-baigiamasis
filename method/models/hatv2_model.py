import torch
from torch import amp
from torch.nn.parallel import DataParallel, DistributedDataParallel
from basicsr.models.sr_model import SRModel
from basicsr.utils.registry import MODEL_REGISTRY
from collections import OrderedDict
from torch.nn import functional as F

from basicsr.losses import build_loss
from basicsr.archs import build_network
from basicsr.utils import get_root_logger
from basicsr.losses.loss_util import get_refined_artifact_map


@MODEL_REGISTRY.register()
class HATModelV2(SRModel):
    """HAT model without GAN training or higher-order degradations.

    Accepts paired LQ/GT input and trains with pixel, perceptual,
    AESOP, TopIQ, and LDL losses. Supports AMP and torch.compile.
    """

    def __init__(self, opt):
        super(HATModelV2, self).__init__(opt)

        if self.is_train:
            self.use_amp = self.opt['train'].get('use_amp', False)
            if self.use_amp:
                self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                self.scaler_g = amp.GradScaler('cuda', enabled=(self.amp_dtype == torch.float16))
                print(f"AMP enabled with dtype: {self.amp_dtype}")
        else:
            self.use_amp = False

    def init_training_settings(self):
        train_opt = self.opt['train']

        # EMA
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)

            for p in self.net_g_ema.parameters():
                p.requires_grad = False

            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        self.net_g.train()

        # losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('ldl_opt'):
            self.cri_ldl = build_loss(train_opt['ldl_opt']).to(self.device)
        else:
            self.cri_ldl = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('topiq_opt'):
            print("Using TopiQ loss...")
            self.cri_topiq = build_loss(train_opt['topiq_opt']).to(self.device)
        else:
            self.cri_topiq = None

        if train_opt.get('aesop_opt'):
            print("Using AESOP loss...")
            self.cri_aesop = build_loss(train_opt['aesop_opt']).to(self.device)
        else:
            self.cri_aesop = None

        # optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

        # TF32 precision for Ampere GPUs
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            torch.set_float32_matmul_precision('high')
            try:
                if hasattr(torch.backends.cudnn, 'conv'):
                    torch.backends.cudnn.conv.fp32_precision = 'tf32'
            except Exception:
                pass

        # torch.compile
        if train_opt.get('use_compile', False):
            if hasattr(torch, 'compile'):
                print("Applying torch.compile to net_g...")
                self.net_g = torch.compile(self.net_g)
            else:
                print("torch.compile is not supported in this version of PyTorch. Skipping...")

    def get_bare_model(self, net):
        """Get bare model, especially under wrapping with
        DistributedDataParallel, DataParallel or torch.compile.
        """
        while isinstance(net, (DataParallel, DistributedDataParallel)) or hasattr(net, '_orig_mod'):
            if isinstance(net, (DataParallel, DistributedDataParallel)):
                net = net.module
            elif hasattr(net, '_orig_mod'):
                net = net._orig_mod
        return net

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        if hasattr(self, "net_g_ema"):
            for p in self.net_g_ema.parameters():
                p.requires_grad = False

        self.optimizer_g.zero_grad()
        with amp.autocast('cuda', enabled=self.use_amp, dtype=getattr(self, 'amp_dtype', torch.float16)):
            self.output = self.net_g(self.lq)

        if self.cri_ldl:
            self.output_ema = self.net_g_ema(self.lq)

        l_g_total = 0
        loss_dict = OrderedDict()

        with amp.autocast('cuda', enabled=self.use_amp, dtype=getattr(self, 'amp_dtype', torch.float16)):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, self.gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            # AESOP loss
            if self.cri_aesop:
                l_g_aesop = self.cri_aesop(self.output, self.gt)
                l_g_total += l_g_aesop
                loss_dict['l_g_aesop'] = l_g_aesop
            # LDL loss
            if self.cri_ldl:
                pixel_weight = get_refined_artifact_map(self.gt, self.output, self.output_ema, 7)
                l_g_ldl = self.cri_ldl(torch.mul(pixel_weight, self.output), torch.mul(pixel_weight, self.gt))
                l_g_total += l_g_ldl
                loss_dict['l_g_ldl'] = l_g_ldl
            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, self.gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style
            # TopIQ loss
            if self.cri_topiq:
                l_g_topiq, _ = self.cri_topiq(self.output, self.gt)
                if l_g_topiq is not None:
                    l_g_total += l_g_topiq
                    loss_dict['l_g_topiq'] = l_g_topiq

        if self.use_amp:
            self.scaler_g.scale(l_g_total).backward()
            self.scaler_g.step(self.optimizer_g)
            self.scaler_g.update()
        else:
            l_g_total.backward()
            self.optimizer_g.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        # pad to multiplication of window_size
        window_size = self.opt['network_g']['window_size']
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(img)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(img)
            self.net_g.train()

        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        self.is_train = False
        super(HATModelV2, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True
