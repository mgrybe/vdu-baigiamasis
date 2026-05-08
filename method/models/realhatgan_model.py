import numpy as np
import random
import torch
from torch import amp
from torch.nn.parallel import DataParallel, DistributedDataParallel
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.models.srgan_model import SRGANModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from collections import OrderedDict
from torch.nn import functional as F

from basicsr.losses import build_loss
from basicsr.archs import build_network
from basicsr.utils import get_root_logger
from basicsr.losses.loss_util import get_refined_artifact_map


@MODEL_REGISTRY.register()
class RealHATGANModel(SRGANModel):
    """GAN-based Real_HAT Model.

    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    def __init__(self, opt):
        super(RealHATGANModel, self).__init__(opt)
        self.jpeger = DiffJPEG(differentiable=False).to(self.device)
        self.usm_sharpener = USMSharp().to(self.device)
        self.queue_size = opt.get('queue_size', 180)

        if self.is_train:
            self.use_amp = self.opt['train'].get('use_amp', False)
            if self.use_amp:
                self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                self.scaler_g = amp.GradScaler('cuda', enabled=(self.amp_dtype == torch.float16))
                self.scaler_d = amp.GradScaler('cuda', enabled=(self.amp_dtype == torch.float16))
                print(f"AMP enabled with dtype: {self.amp_dtype}")
        else:
            self.use_amp = False

    def init_training_settings(self):
        # 1. Call super to initialize networks, losses, and optimizers
        #super(RealHATGANModel, self).init_training_settings()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)

            # Added by me: https://github.com/2minkyulee/AESOP-Auto-Encoded-Supervision-for-Perceptual-Image-Super-Resolution/blob/25115ea9cfb14e2e74c5963e8d5ef09342914920/AESOP/basicsr/models/aesop_esrganArtifactsDis_model.py#L22C1-L23C40
            for p in self.net_g_ema.parameters():
                p.requires_grad = False

            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define network net_d
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True), param_key)

        self.net_g.train()
        self.net_d.train()

        # define losses
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

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

        # MY PART BEGINS HERE
        #train_opt = self.opt['train']

        if train_opt.get('topiq_opt'):
            print("Using TopiQ loss...")
            self.cri_topiq = build_loss(train_opt['topiq_opt']).to(self.device)
        else:
            self.cri_topiq = None

        if train_opt.get('aesop_opt'):
            print("Using AESOP loss...")
            self.cri_aesop = build_loss(train_opt['aesop_opt']).to(self.device)
        else:
            # raise ValueError('aesop_opt must be specified.')
            self.cri_aesop = None

        # if train_opt.get('artifacts_opt'):
        #     self.cri_artifacts = build_loss(train_opt['artifacts_opt']).to(self.device)
        # else:
        #     self.cri_artifacts = None

        # 2. Set matmul and conv precision for Ampere GPUs
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            # Modern API for TF32 (silences warnings in PyTorch 2.5+)
            torch.set_float32_matmul_precision('high')
            try:
                if hasattr(torch.backends.cudnn, 'conv'):
                    torch.backends.cudnn.conv.fp32_precision = 'tf32'
            except Exception:
                pass

        # self.ema_decay = train_opt.get('ema_decay', 0)
        # if self.ema_decay > 0:
        #     logger = get_root_logger()
        #     logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
        #     self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
        #     for p in self.net_g_ema.parameters():
        #         p.requires_grad = False
        #     # load pretrained model
        #     load_path = self.opt['path'].get('pretrain_network_g', None)
        #     if load_path is not None:
        #         self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
        #     else:
        #         self.model_ema(0)  # copy net_g weight
        #     self.net_g_ema.eval()

        # 3. Apply torch.compile if enabled in config
        if train_opt.get('use_compile', False):
            if hasattr(torch, 'compile'):
                print("Applying torch.compile to net_g and net_d...")
                # You can compile the generator
                self.net_g = torch.compile(self.net_g)
                # And the discriminator
                self.net_d = torch.compile(self.net_d)
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

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def feed_data(self, data):
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images.
        """
        if self.is_train and self.opt.get('high_order_degradation', True):
            # training data synthesis
            self.gt = data['gt'].to(self.device)
            self.gt_usm = self.usm_sharpener(self.gt)

            self.kernel1 = data['kernel1'].to(self.device)
            self.kernel2 = data['kernel2'].to(self.device)
            self.sinc_kernel = data['sinc_kernel'].to(self.device)

            ori_h, ori_w = self.gt.size()[2:4]

            # ----------------------- The first degradation process ----------------------- #
            # blur
            out = filter2D(self.gt_usm, self.kernel1)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['resize_range'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['resize_range'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, scale_factor=scale, mode=mode)
            # add noise
            gray_noise_prob = self.opt['gray_noise_prob']
            if np.random.uniform() < self.opt['gaussian_noise_prob']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt['poisson_scale_range'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
            out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
            out = self.jpeger(out, quality=jpeg_p)

            # ----------------------- The second degradation process ----------------------- #
            # blur
            if np.random.uniform() < self.opt['second_blur_prob']:
                out = filter2D(out, self.kernel2)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['resize_range2'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['resize_range2'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                out, size=(int(ori_h / self.opt['scale'] * scale), int(ori_w / self.opt['scale'] * scale)), mode=mode)
            # add noise
            gray_noise_prob = self.opt['gray_noise_prob2']
            if np.random.uniform() < self.opt['gaussian_noise_prob2']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt['poisson_scale_range2'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)

            # JPEG compression + the final sinc filter
            # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
            # as one operation.
            # We consider two orders:
            #   1. [resize back + sinc filter] + JPEG compression
            #   2. JPEG compression + [resize back + sinc filter]
            # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
            if np.random.uniform() < 0.5:
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
            else:
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)

            # clamp and round
            self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

            # random crop
            gt_size = self.opt['gt_size']
            (self.gt, self.gt_usm), self.lq = paired_random_crop([self.gt, self.gt_usm], self.lq, gt_size,
                                                                 self.opt['scale'])

            # training pair pool
            self._dequeue_and_enqueue()
            # sharpen self.gt again, as we have changed the self.gt with self._dequeue_and_enqueue
            self.gt_usm = self.usm_sharpener(self.gt)
            self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract
        else:
            # for paired training or validation
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)
                self.gt_usm = self.usm_sharpener(self.gt)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        super(RealHATGANModel, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True

    def optimize_parameters(self, current_iter):
        # usm sharpening
        l1_gt = self.gt_usm
        percep_gt = self.gt_usm
        gan_gt = self.gt_usm
        if self.opt['l1_gt_usm'] is False:
            l1_gt = self.gt
        if self.opt['percep_gt_usm'] is False:
            percep_gt = self.gt
        if self.opt['gan_gt_usm'] is False:
            gan_gt = self.gt

        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        # Added by me: https://github.com/2minkyulee/AESOP-Auto-Encoded-Supervision-for-Perceptual-Image-Super-Resolution/blob/25115ea9cfb14e2e74c5963e8d5ef09342914920/AESOP/basicsr/models/aesop_esrganArtifactsDis_model.py#L87C1-L89C40
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
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            with amp.autocast('cuda', enabled=self.use_amp, dtype=getattr(self, 'amp_dtype', torch.float16)):
                # pixel loss
                if self.cri_pix:
                    l_g_pix = self.cri_pix(self.output, l1_gt)
                    l_g_total += l_g_pix
                    loss_dict['l_g_pix'] = l_g_pix
                if self.cri_aesop:
                    l_g_aesop = self.cri_aesop(self.output, l1_gt)
                    l_g_total += l_g_aesop
                    loss_dict['l_g_aesop'] = l_g_aesop
                if self.cri_ldl:
                    pixel_weight = get_refined_artifact_map(self.gt, self.output, self.output_ema, 7)
                    l_g_ldl = self.cri_ldl(torch.mul(pixel_weight, self.output), torch.mul(pixel_weight, self.gt))
                    l_g_total += l_g_ldl
                    loss_dict['l_g_ldl'] = l_g_ldl
                # if self.cri_artifacts:
                #     # https://github.com/2minkyulee/AESOP-Auto-Encoded-Supervision-for-Perceptual-Image-Super-Resolution/blob/25115ea9cfb14e2e74c5963e8d5ef09342914920/AESOP/basicsr/models/aesop_esrganArtifactsDis_model.py#L51
                #     pixel_weight = get_refined_artifact_map(self.gt, self.output, self.output_ema, 7)
                #     l_g_artifacts = self.cri_artifacts(torch.mul(pixel_weight, self.output), torch.mul(pixel_weight, self.gt))
                #     l_g_total += l_g_artifacts
                #     loss_dict['l_g_artifacts'] = l_g_artifacts
                # perceptual loss
                if self.cri_perceptual:
                    l_g_percep, l_g_style = self.cri_perceptual(self.output, percep_gt)
                    if l_g_percep is not None:
                        l_g_total += l_g_percep
                        loss_dict['l_g_percep'] = l_g_percep
                    if l_g_style is not None:
                        l_g_total += l_g_style
                        loss_dict['l_g_style'] = l_g_style
                # topiq loss
                if self.cri_topiq:
                    l_g_percep, l_g_style = self.cri_topiq(self.output, self.gt)
                    if l_g_percep is not None:
                        l_g_total += l_g_percep
                        loss_dict['l_g_topiq'] = l_g_percep
                    # if l_g_style is not None:
                    #     l_g_total += l_g_style
                    #     loss_dict['l_g_style'] = l_g_style
                # gan loss
                fake_g_pred = self.net_d(self.output)
                l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
                l_g_total += l_g_gan
                loss_dict['l_g_gan'] = l_g_gan

            if self.use_amp:
                self.scaler_g.scale(l_g_total).backward()
                self.scaler_g.step(self.optimizer_g)
                self.scaler_g.update()
            else:
                l_g_total.backward()
                self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # real
        with amp.autocast('cuda', enabled=self.use_amp, dtype=getattr(self, 'amp_dtype', torch.float16)):
            real_d_pred = self.net_d(gan_gt)
            l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
            loss_dict['l_d_real'] = l_d_real
            loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        if self.use_amp:
            self.scaler_d.scale(l_d_real).backward()
        else:
            l_d_real.backward()
        # fake
        with amp.autocast('cuda', enabled=self.use_amp, dtype=getattr(self, 'amp_dtype', torch.float16)):
            fake_d_pred = self.net_d(self.output.detach().clone())  # clone for pt1.9
            l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
            loss_dict['l_d_fake'] = l_d_fake
            loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        if self.use_amp:
            self.scaler_d.scale(l_d_fake).backward()
            self.scaler_d.step(self.optimizer_d)
            self.scaler_d.update()
        else:
            l_d_fake.backward()
            self.optimizer_d.step()

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