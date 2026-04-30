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
class AesopESRGANArtifactsDisMoEModel(SRGANModel):

    def __init__(self, opt):
        super(AesopESRGANArtifactsDisMoEModel, self).__init__(opt)
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


        # Tau annealing for Gumbel-Softmax routing in MOD discriminator.
        # Tau controls gradient sharpness through the straight-through estimator.
        # High tau (1.0) = diffuse gradients, good for early exploration.
        # Low tau (0.1) = sharp gradients, drives logit margin growth.
        tau_cfg = train_opt.get('tau_anneal', {})
        self.tau_start = tau_cfg.get('tau_start', 1.0)
        self.tau_end = tau_cfg.get('tau_end', 0.1)
        self.tau_anneal_iters = tau_cfg.get('anneal_iters', 200000)

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
        # Linearly scale progress from 0.0 to 1.0
        progress = min(current_iter / self.tau_anneal_iters, 1.0)

        # Exponentially anneal from start to end
        # Formula: tau_t = start * (end/start) ^ progress
        current_tau = self.tau_start * (self.tau_end / self.tau_start) ** progress

        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        if hasattr(self, "net_g_ema"):
            for p in self.net_g_ema.parameters():
                p.requires_grad = False

        self.optimizer_g.zero_grad()
        with amp.autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
            self.output = self.net_g(self.lq)

        l_g_total = 0
        loss_dict = OrderedDict()

        with amp.autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
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
            if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
                # gan loss (relativistic gan)
                real_d_pred, routing, _, _ = self.net_d(self.gt, tau=current_tau)
                real_d_pred = real_d_pred.detach()
                fake_g_pred, _, _, _ = self.net_d(self.output, routing.detach(), tau=current_tau)

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
        with amp.autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
            real_d_pred, routing, features, ortho_w = self.net_d(self.gt, tau=current_tau)
            fake_d_pred, _, _, _ = self.net_d(self.output, routing.detach(), tau=current_tau)
            fake_d_pred = fake_d_pred.detach()

            l_d_real = self.cri_gan(
                real_d_pred - torch.mean(fake_d_pred), True, is_disc=True
            ) * 0.5

            # Eq. 6: orthogonal loss (λ_o = 10)
            l_d_ortho = self.orthogonal_loss(ortho_w) * 0.1 # 10.0
            l_d_real += l_d_ortho
            loss_dict['l_d_ortho'] = l_d_ortho

            # Eq. 7: LDA distribution loss (λ_d = 10)
            l_d_dist = self.distribution_loss(features) * 0.1 #10.0
            l_d_real += l_d_dist
            loss_dict['l_d_dist'] = l_d_dist

            # Eq. 4: balancing loss (λ_b = 0.05) — uses soft R' from HR call
            l_d_balance = self.balancing_loss(self.net_d._R_prime, self.net_d.num_expert) * 0.00024 # 0.05
            l_d_real += l_d_balance
            loss_dict['l_d_balance'] = l_d_balance

            with torch.no_grad():
                # ── Routing diagnostics ─────────────────────────────────────
                import math
                N = self.net_d.num_expert
                probs = self.net_d._R_prime.detach()
                hard_routing = routing.detach().argmax(dim=1)

                # ==========================================================
                # 1. Router Confidence (Per-pixel entropy)
                # ==========================================================
                # What it measures: How certain the router is about individual pixels.
                # 🟢 GOOD: Drops toward 0.0 as training progresses (highly confident).
                # 🔴 BAD: Stays near math.log(N) (e.g., 2.48 for 12 experts). Means
                #         the router is outputting flat, uncertain probabilities.
                pixel_entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
                loss_dict['routing_pixel_entropy'] = pixel_entropy

                # ==========================================================
                # 2. Load Balancing (Marginal entropy & Usage)
                # ==========================================================
                # What it measures: Is the overall workload distributed fairly?
                expert_usage = routing.detach().float().mean(dim=(0, 2, 3))  # (N,)
                marginal_entropy = -torch.sum(expert_usage * torch.log(expert_usage + 1e-8))

                # Marginal Ratio normalizes the entropy between 0.0 and 1.0
                # 🟢 GOOD: Hovers between 0.8 and 1.0. All experts are employed.
                # 🟡 WARNING: Drops to 0.4 - 0.6. Some experts are dying.
                # 🔴 DANGER: Drops below 0.2. Total routing collapse (1 expert took over).
                loss_dict['routing_marginal_entropy'] = marginal_entropy
                loss_dict['routing_marginal_ratio'] = marginal_entropy / math.log(N)

                # Usage Stats (Fractions between 0.0 and 1.0)
                # 🟢 GOOD max: Close to (1.0 / N) (e.g., ~0.08 for 12 experts).
                # 🔴 BAD max: Approaching 1.0 (One expert is hoarding all pixels).
                loss_dict['routing_usage_std'] = expert_usage.std()
                loss_dict['routing_usage_max'] = expert_usage.max()

                # ==========================================================
                # 3. Spatial Fragmentation (Clumping vs. White Noise)
                # ==========================================================
                # What it measures: What percentage of pixels are assigned to a
                # DIFFERENT expert than their immediate neighbor?
                diff_x = (hard_routing[:, :, 1:] != hard_routing[:, :, :-1]).float().mean()
                diff_y = (hard_routing[:, 1:, :] != hard_routing[:, :-1, :]).float().mean()
                fragmentation = (diff_x + diff_y) / 2.0

                # 🟢 GOOD: Slowly drops toward < 0.15 (15%). Experts are claiming
                #          solid, contiguous chunks of texture or structure.
                # 🔴 BAD: Pinned constantly > 0.50 (50%). You have static white noise.
                #         (Action: Cut your balancing loss weight drastically).
                loss_dict['routing_fragmentation'] = fragmentation

                # ==========================================================
                # 4. Gumbel Hardness (Logit Margin)
                # ==========================================================
                # What it measures: The raw mathematical distance between the router's
                # #1 choice and #2 choice BEFORE any softmax or noise is applied.
                F_feat = self.net_d.FE(self.gt)
                routing_logits = self.net_d.router(F_feat)  # (B, N, H/4, W/4)
                top2 = routing_logits.topk(2, dim=1).values
                margin = (top2[:, 0] - top2[:, 1]).mean()

                # 🟢 GOOD: Grows steadily over time, eventually exceeding 2.0 or 3.0.
                #          This means the router's choice is bulletproof against the
                #          random Gumbel noise injected during training.
                # 🔴 BAD: Stays near 0.0. This means choices are tied, and the Gumbel
                #         noise is randomly deciding where pixels go.
                loss_dict['routing_margin'] = margin
        if self.use_amp:
            self.scaler_d.scale(l_d_real).backward()
        else:
            l_d_real.backward()

        # fake
        with amp.autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
            fake_d_pred, _, _, _ = self.net_d(self.output.detach(), routing.detach(), tau=current_tau)
            l_d_fake = self.cri_gan(
                fake_d_pred - torch.mean(real_d_pred.detach()), False, is_disc=True
            ) * 0.5
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
        self.log_dict['tau'] = current_tau

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    # ═══════════════════════════════════════════════════════════════════════════
    # Loss functions (Eqs. 4, 6, 7)
    #
    # Designed to be used as methods on ESRGANModel or standalone functions
    # called during discriminator optimization.
    # ═══════════════════════════════════════════════════════════════════════════

    def balancing_loss(self, R_prime, num_expert):
        """
        Eq. 4: L_b = (N / M) Σ_{i,j} max_k R'[i, j, k]

        Penalizes peaked softmax distributions to enforce uniform expert usage.

        Args:
            R_prime: Soft routing from softmax (B, N, H, W).
            num_expert: N.
        """
        N = num_expert
        max_vals = R_prime.max(dim=1)[0]  # (B, H, W)
        M = max_vals.shape[1] * max_vals.shape[2]
        L_b = (N / M) * max_vals.sum(dim=(1, 2)).mean()
        return L_b


    def orthogonal_loss(self, ortho_weights):
        """
        Eq. 6: L_o = ||ψ(O)ψ(O)^T − I_N||²_F

        Forces N orthogonal conv weights to be mutually orthogonal.

        Args:
            ortho_weights: Stacked weights (N, c, c, 1, 1).
        """
        N = ortho_weights.shape[0]
        psi = ortho_weights.view(N, -1)       # (N, c²)
        gram = psi @ psi.T                    # (N, N)
        I_N = torch.eye(N, device=gram.device)
        L_o = (gram - I_N).pow(2).sum()
        return L_o


    def distribution_loss(self, F_stars):
        """
        Eq. 7: L_dist = L_btw + L_wi

        LDA-inspired: maximize between-class separation, minimize within-class
        variance.

        Args:
            F_stars: List of N tensors, each (B, c, H, W).
        """
        N = len(F_stars)
        means = [f.mean(dim=(2, 3)) for f in F_stars]  # each (B, c)

        # L_btw: squared cosine similarity between all class-mean pairs
        L_btw = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                cos_sim = F.cosine_similarity(means[i], means[j], dim=1)
                L_btw = L_btw + (cos_sim ** 2).mean()

        # L_wi: total spatial variance across all classes
        L_wi = sum(f.var(dim=(2, 3)).mean() for f in F_stars)

        return L_btw + L_wi