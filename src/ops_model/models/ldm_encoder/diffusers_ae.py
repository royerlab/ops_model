# %%
import functools
from typing import Optional, Union

import pytorch_lightning as L
import torch
import torch.nn as nn
import torchvision.utils as tvutils
import torch.nn.functional as F


class LitAe(L.LightningModule):
    def __init__(
        self,
        model,
        model_kwargs: Optional[dict] = None,
        lightning_config: Optional[dict] = None,
    ):
        super().__init__()
        self.model = model(**(model_kwargs or {}))
        self.gen_lr = lightning_config.get("gen_lr", 4.5e-6)
        self.disc_lr = lightning_config.get("disc_lr", self.gen_lr)
        self.sample_posterior = False
        self.eps = 1e-8

        #
        self.logvar_init = 0.0
        self.logvar = nn.Parameter(torch.ones(size=()) * self.logvar_init)
        self.kl_weight = lightning_config["kl_weight"]

        # GAN parts
        self.last_layer = [self._get_last_layer_param()]
        self.discriminator = NLayerDiscriminator(
            input_nc=self.model.config.out_channels,
            ndf=64,
            n_layers=3,
        )
        self.disc_loss = hinge_d_loss
        self.disc_factor = lightning_config.get("disc_factor", 1.0)
        self.disc_iter_start = lightning_config.get("disc_iter_start", 10001)
        self.discriminator_weight = lightning_config.get("discriminator_weight", 1.0)

        self.automatic_optimization = False

    def forward(self, x):
        posterior = self.model.encode(x).latent_dist
        if self.sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        x_rec = self.model.decode(z).sample
        return x_rec, posterior

    def training_step(self, batch):
        inputs = batch["data"]

        outputs, posterior = self(inputs)

        g_opt, d_opt = self.optimizers()

        # ==== Generator Update ====
        self.toggle_optimizer(g_opt)

        # Reconstruction Loss
        rec_loss = F.mse_loss(
            outputs, inputs
        )  # can replace with L1 for sharper reconstructions
        nll_loss = (
            rec_loss / torch.exp(self.logvar) + self.logvar
        )  # can add weights if needed

        # KL Loss
        kl_loss = posterior.kl().mean()

        # GAN generator Loss
        logits_fake = self.discriminator(outputs)
        g_loss = -torch.mean(logits_fake)

        disc_factor = adopt_weight(
            self.disc_factor, self.global_step, threshold=self.disc_iter_start
        )
        if disc_factor > 0.0:
            try:
                d_weight = self.calculate_adaptive_weight(
                    nll_loss, g_loss, last_layer=None
                )
            except RuntimeError:
                d_weight = torch.tensor(0.0)
        else:
            d_weight = torch.tensor(0.0)

        loss = nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss

        self.manual_backward(loss)
        g_opt.step()
        g_opt.zero_grad()
        self.untoggle_optimizer(g_opt)

        # ===== Discriminator Update =====
        # GAN discriminator Loss
        if disc_factor > 0.0:
            self.toggle_optimizer(d_opt)
            logits_real = self.discriminator(inputs.detach())
            logits_fake = self.discriminator(outputs.detach())
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            self.manual_backward(d_loss)
            d_opt.step()
            d_opt.zero_grad()
            self.untoggle_optimizer(d_opt)

        else:
            d_loss = torch.tensor(0.0)

        self.log_dict(
            {
                "train/nll_loss": nll_loss.detach(),
                "train/kl_loss": kl_loss.detach(),
                "train/g_loss": g_loss.detach(),
                "train/d_loss": d_loss.detach(),
                "train/d_weight": d_weight.detach(),
                "train/disc_factor": torch.as_tensor(disc_factor, device=self.device),
                "train/logvar": self.logvar.detach(),
            },
            on_step=True,
            prog_bar=False,
            batch_size=inputs.size(0),
        )

        self.log(
            "train/loss",
            loss.detach(),
            on_step=True,
            prog_bar=True,
            batch_size=inputs.size(0),
        )

        return None  # manual optimization in calc_losses

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        inputs = batch["data"]

        outputs, posterior = self(inputs)

        rec_loss = F.mse_loss(
            outputs, inputs
        )  # can replace with L1 for sharper reconstructions
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar  #
        kl_loss = posterior.kl().mean()

        loss_total = nll_loss + self.kl_weight * kl_loss

        self.log_dict(
            {
                "val/nll_loss": nll_loss.detach(),
                "val/kl_loss": kl_loss.detach(),
                "val/loss": loss_total.detach(),
                "val/logvar": self.logvar.detach(),
            },
            prog_bar=True,
            batch_size=inputs.size(0),
        )

        if batch_idx == 0:
            self.log_images(inputs, outputs)

        return None

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.model.parameters(), lr=self.gen_lr)
        d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.disc_lr)
        return [g_opt, d_opt]

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(
                nll_loss, self.last_layer[0], retain_graph=True
            )[0]
            g_grads = torch.autograd.grad(
                g_loss, self.last_layer[0], retain_graph=True
            )[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def _get_last_layer_param(self):
        # pick a stable leaf param from the decoder (e.g., last conv weight)
        # adjust to your model structure:
        for n, p in reversed(list(self.model.decoder.named_parameters())):
            if p.requires_grad and p.ndim >= 2:
                return p
        # fallback: first trainable param
        return next(self.model.decoder.parameters())

    def log_images(self, inputs, outputs, num_samples=8):
        n = min(inputs.size(0), num_samples)
        grid = tvutils.make_grid(
            torch.cat((inputs[:n], outputs[:n])),
            nrow=n,
            normalize=True,
            value_range=(-1, 1),
        )
        self.logger.experiment.add_image("reconstructions", grid, self.current_epoch)


class NLayerDiscriminator(nn.Module):
    # Adapted from Taming-transformers
    """Defines a PatchGAN discriminator as in Pix2Pix
    --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        norm_layer = nn.BatchNorm2d
        if (
            type(norm_layer) == functools.partial
        ):  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def adopt_weight(weight, global_step, threshold=0, value=0.0):
    if global_step < threshold:
        weight = value
    return weight
