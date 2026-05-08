import torch
from torch import amp
from torch.nn import functional as F
from collections import OrderedDict

from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.srgan_model import SRGANModel
from basicsr.losses import build_loss
from basicsr.archs import build_network
from basicsr.utils import get_root_logger

@MODEL_REGISTRY.register()
class AesopESRGANArtifactsDisModel(SRGANModel):

    def __init__(self, opt):
        super(AesopESRGANArtifactsDisModel, self).__init__(opt)
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
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
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

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_d', None)
        load_key = self.opt['path'].get('param_key_g', None)
        if load_path is not None:
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True), load_key)

        self.net_g.train()
        self.net_d.train()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('artifacts_opt'):
            self.cri_artifacts = build_loss(train_opt['artifacts_opt']).to(self.device)
        else:
            self.cri_artifacts = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)
        else:
            raise ValueError('gan_opt must be specified.')
        
        if train_opt.get('aesop_opt'):
            self.cri_aesop = build_loss(train_opt['aesop_opt']).to(self.device)
        else:
            # raise ValueError('aesop_opt must be specified.')
            self.cri_aesop = None
        

        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

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

    def optimize_parameters(self, current_iter):
        
        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False
        
        if hasattr(self, "net_g_ema"):
            for p in self.net_g_ema.parameters():
                p.requires_grad = False

        self.optimizer_g.zero_grad()
        with amp.autocast('cuda', enabled=self.use_amp, dtype=getattr(self, 'amp_dtype', torch.float16)):
            self.output = self.net_g(self.lq)

        if hasattr(self, "net_g_ema"):
            self.output_ema = self.net_g_ema(self.lq)

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            with amp.autocast('cuda', enabled=self.use_amp, dtype=getattr(self, 'amp_dtype', torch.float16)):
                # pixel loss
                if self.cri_pix:
                    l_g_pix = self.cri_pix(self.output, self.gt)
                    l_g_total += l_g_pix
                    loss_dict['l_g_pix'] = l_g_pix

                if self.cri_aesop:
                    l_g_aesep = self.cri_aesop(self.output, self.gt)
                    l_g_total += l_g_aesep
                    loss_dict['l_g_aesep'] = l_g_aesep

                # perceptual loss
                if self.cri_perceptual:
                    l_g_percep, l_g_style = self.cri_perceptual(self.output, self.gt)
                    if l_g_percep is not None:
                        l_g_total += l_g_percep
                        loss_dict['l_g_percep'] = l_g_percep
                    if l_g_style is not None:
                        l_g_total += l_g_style
                        loss_dict['l_g_style'] = l_g_style
                # gan loss (relativistic gan)
                real_d_pred = self.net_d(self.gt).detach()
                fake_g_pred = self.net_d(self.output)
                l_g_real = self.cri_gan(real_d_pred - torch.mean(fake_g_pred), False, is_disc=False)
                l_g_fake = self.cri_gan(fake_g_pred - torch.mean(real_d_pred), True, is_disc=False)
                l_g_gan = (l_g_real + l_g_fake) / 2

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
            fake_d_pred = self.net_d(self.output).detach()
            real_d_pred = self.net_d(self.gt)
            l_d_real = self.cri_gan(real_d_pred - torch.mean(fake_d_pred), True, is_disc=True) * 0.5
        if self.use_amp:
            self.scaler_d.scale(l_d_real).backward()
        else:
            l_d_real.backward()

        # fake
        with amp.autocast('cuda', enabled=self.use_amp, dtype=getattr(self, 'amp_dtype', torch.float16)):
            fake_d_pred = self.net_d(self.output.detach())
            l_d_fake = self.cri_gan(fake_d_pred - torch.mean(real_d_pred.detach()), False, is_disc=True) * 0.5
        if self.use_amp:
            self.scaler_d.scale(l_d_fake).backward()
            self.scaler_d.step(self.optimizer_d)
            self.scaler_d.update()
        else:
            l_d_fake.backward()
            self.optimizer_d.step()

        loss_dict['l_d_real'] = l_d_real
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)


@MODEL_REGISTRY.register()
class ProbabilisticAesopESRGANArtifactsDisModel(SRGANModel):

    def __init__(self, opt):
        super(ProbabilisticAesopESRGANArtifactsDisModel, self).__init__(opt)
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
        train_opt = self.opt['train']
        
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
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
        
        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_d', None)
        load_key = self.opt['path'].get('param_key_g', None)
        if load_path is not None:
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True), load_key)
        
        self.net_g.train()
        self.net_d.train()
        
        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None
        
        if train_opt.get('artifacts_opt'):
            self.cri_artifacts = build_loss(train_opt['artifacts_opt']).to(self.device)
        else:
            self.cri_artifacts = None
        
        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None
        
        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)
        else:
            raise ValueError('gan_opt must be specified.')
        
        if train_opt.get('aesop_opt'):
            raise NotImplementedError("Dont use aesop loss for now. Instead use prob_aesop loss.")
        
        if train_opt.get('prob_aesop_opt'):
            self.cri_prob_aesop = build_loss(train_opt['prob_aesop_opt']).to(self.device)
        else:
            # raise ValueError('prob_aesop_opt must be specified.')
            self.cri_prob_aesop = None
        
        
        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)
        
        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
    
    def optimize_parameters(self, current_iter):
        
        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False
        
        if hasattr(self, "net_g_ema"):
            for p in self.net_g_ema.parameters():
                p.requires_grad = False
        
        self.optimizer_g.zero_grad()
        with amp.autocast('cuda', enabled=self.use_amp, dtype=getattr(self, 'amp_dtype', torch.float16)):
            self.output = self.net_g(self.lq)

        if hasattr(self, "net_g_ema"):
            self.output_ema = self.net_g_ema(self.lq)

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            with amp.autocast('cuda', enabled=self.use_amp, dtype=getattr(self, 'amp_dtype', torch.float16)):
                # pixel loss
                if self.cri_pix:
                    l_g_pix = self.cri_pix(self.output, self.gt)
                    l_g_total += l_g_pix
                    loss_dict['l_g_pix'] = l_g_pix

                if hasattr(self, "cri_aesop"):
                    raise NotImplementedError("Dont use aesop loss for now. Instead use prob_aesop loss.")

                if self.cri_prob_aesop:
                    l_g_prob_aesep = self.cri_prob_aesop(self.output, self.gt)
                    l_g_total += l_g_prob_aesep
                    loss_dict['l_g_prob_aesep'] = l_g_prob_aesep

                # perceptual loss
                if self.cri_perceptual:
                    l_g_percep, l_g_style = self.cri_perceptual(self.output, self.gt)
                    if l_g_percep is not None:
                        l_g_total += l_g_percep
                        loss_dict['l_g_percep'] = l_g_percep
                    if l_g_style is not None:
                        l_g_total += l_g_style
                        loss_dict['l_g_style'] = l_g_style
                # gan loss (relativistic gan)
                real_d_pred = self.net_d(self.gt).detach()
                fake_g_pred = self.net_d(self.output)
                l_g_real = self.cri_gan(real_d_pred - torch.mean(fake_g_pred), False, is_disc=False)
                l_g_fake = self.cri_gan(fake_g_pred - torch.mean(real_d_pred), True, is_disc=False)
                l_g_gan = (l_g_real + l_g_fake) / 2

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
            fake_d_pred = self.net_d(self.output).detach()
            real_d_pred = self.net_d(self.gt)
            l_d_real = self.cri_gan(real_d_pred - torch.mean(fake_d_pred), True, is_disc=True) * 0.5
        if self.use_amp:
            self.scaler_d.scale(l_d_real).backward()
        else:
            l_d_real.backward()
        # fake
        with amp.autocast('cuda', enabled=self.use_amp, dtype=getattr(self, 'amp_dtype', torch.float16)):
            fake_d_pred = self.net_d(self.output.detach())
            l_d_fake = self.cri_gan(fake_d_pred - torch.mean(real_d_pred.detach()), False, is_disc=True) * 0.5
        if self.use_amp:
            self.scaler_d.scale(l_d_fake).backward()
            self.scaler_d.step(self.optimizer_d)
            self.scaler_d.update()
        else:
            l_d_fake.backward()
            self.optimizer_d.step()

        loss_dict['l_d_real'] = l_d_real
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        
        self.log_dict = self.reduce_loss_dict(loss_dict)
        
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
