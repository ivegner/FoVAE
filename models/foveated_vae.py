import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
from torch import nn, optim
from typing import *

import matplotlib.pyplot as plt

from utils.visualization import imshow_unnorm
from utils.foveation import get_gaussian_foveation_filter


def reparam_sample(mu, logvar):
    std = torch.exp(0.5 * logvar)  # e^(1/2 * log(std^2))
    eps = torch.randn_like(std)  # random ~ N(0, 1)
    return mu + std * eps


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class UpBlock(nn.Module):
    def __init__(self, from_sample_dim, to_z_dim, hidden_ff_out_dims=None):
        super().__init__()
        self.out_z_dim = to_z_dim

        if not hidden_ff_out_dims:
            hidden_ff_out_dims = []

        # if no hidden dims provided, will contain just z dim (*2 bc mean and logvar)
        hidden_ff_out_dims = [*hidden_ff_out_dims, to_z_dim * 2]

        stack = []
        last_out_dim = from_sample_dim
        for nn_out_dim in hidden_ff_out_dims:
            stack.extend([nn.GELU(), nn.Linear(last_out_dim, nn_out_dim)])
            last_out_dim = nn_out_dim

        self.encoder = nn.Sequential(View((-1, from_sample_dim)), *stack)

    def forward(self, x):
        distributions = self.encoder(x)
        mu = distributions[:, : self.out_z_dim]
        logvar = distributions[:, self.out_z_dim :]
        return mu, logvar


class DownBlock(nn.Module):
    def __init__(self, from_dim, to_dim, hidden_ff_out_dims=None):
        super().__init__()

        if not hidden_ff_out_dims:
            hidden_ff_out_dims = []

        # if no hidden dims provided, will contain just to_dim
        hidden_ff_out_dims = [*hidden_ff_out_dims, to_dim]

        stack = []
        last_out_dim = from_dim
        for nn_out_dim in hidden_ff_out_dims:
            stack.extend([nn.GELU(), nn.Linear(last_out_dim, nn_out_dim)])
            last_out_dim = nn_out_dim

        self.decoder = nn.Sequential(View((-1, from_dim)), *stack)

    def forward(self, x):
        return self.decoder(x)


# class Sampler(nn.Module):
#     def __init__(self):
#         super.__init__()

#     def forward(self, mu, logvar):
#         return reparam_sample(mu, logvar)


class FoVAE(pl.LightningModule):
    def __init__(
        self,
        image_dim=28,
        fovea_radius=2,
        patch_dim=5,
        patch_channels=3,
        z_dim=10,
        n_steps: int = 1,
        foveation_padding: Union[Literal["max"], int] = 14,#"max",
        foveation_padding_mode: Literal["zeros", "replicate"] = "zeros",
        lr=1e-3,
        beta=1,
        # grad_clip=100,
        grad_skip_threshold=1000,
        do_add_pos_encoding=True,
        do_use_beta_norm=True,
    ):
        super().__init__()

        self.image_dim = image_dim
        self.fovea_radius = fovea_radius
        self.patch_dim = patch_dim
        self.z_dim = z_dim

        self.n_steps = n_steps
        if do_add_pos_encoding:
            self.num_channels = patch_channels + 2
        else:
            self.num_channels = patch_channels
        self.lr = lr
        self.foveation_padding = foveation_padding
        self.foveation_padding_mode = foveation_padding_mode

        input_dim = self.patch_dim * self.patch_dim * self.num_channels

        self.encoders = nn.ModuleList([UpBlock(input_dim, z_dim)])  # , [1024, 256]),
        self.decoders = nn.ModuleList([DownBlock(z_dim, input_dim)])  # , [256, 1024]),

        # self.encoder = nn.Sequential(
        #     View((-1, input_dim)),
        #     nn.GELU(),
        #     nn.Linear(input_dim, 1024),
        #     nn.GELU(),
        #     nn.Linear(1024, 256),
        #     nn.GELU(),
        #     nn.Linear(256, z_dim * 2),
        # )
        # self.decoder = nn.Sequential(
        #     nn.GELU(),
        #     nn.Linear(z_dim, 256),
        #     nn.GELU(),
        #     nn.Linear(256, 1024),
        #     nn.GELU(),
        #     nn.Linear(1024, input_dim),
        #     View((-1, self.num_channels, self.patch_dim, self.patch_dim)),
        # )
        self._beta = beta
        # self.encoder = nn.Sequential(
        #     # nn.Conv2d(3, 32, 4, 2, 1),          # B,  32, 32, 32
        #     # nn.ReLU(True),
        #     nn.Conv2d(3, 32, 4, 2, 1),          # B,  32, 16, 16
        #     nn.ReLU(True),
        #     nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  8,  8
        #     nn.ReLU(True),
        #     nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  4,  4
        #     nn.ReLU(True),
        #     nn.Conv2d(64, 256, 4, 1),            # B, 256,  1,  1
        #     nn.ReLU(True),
        #     View((-1, 256*1*1)),                 # B, 256
        #     nn.Linear(256, z_dim*2),             # B, z_dim*2
        # )
        # self.decoder = nn.Sequential(
        #     nn.Linear(z_dim, 256),               # B, 256
        #     View((-1, 256, 1, 1)),               # B, 256,  1,  1
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(256, 64, 4),      # B,  64,  4,  4
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  64,  8,  8
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  32, 16, 16
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(32, 3, 4, 2, 1), # B,  nc, 32, 32
        #     # nn.ReLU(True),
        #     # nn.ConvTranspose2d(32, 3, 4, 2, 1),  # B, nc, 64, 64
        # )
        # for block in self._modules:
        #     for m in self._modules[block]:
        #         kaiming_init(m)

        # self.grad_clip = grad_clip
        self.grad_skip_threshold = grad_skip_threshold
        self.do_add_pos_encoding = do_add_pos_encoding
        self.do_use_beta_norm = do_use_beta_norm
        if self.do_use_beta_norm:
            self.beta = (beta * z_dim) / input_dim  # according to beta-vae paper
            print(
                f"Using beta_norm value of {self.beta:.6f} as beta, calculated from unnormalized beta {beta:.6f}"
            )
        else:
            self.beta = beta

        # image: (b, c, image_dim[0], image_dim[1])
        # filters: (fov_h, fov_w, image_dim[0], image_dim[1])
        # TODO: sparsify
        self.register_buffer(
            "foveation_filters",
            torch.from_numpy(
                get_gaussian_foveation_filter(
                    image_dim=(image_dim, image_dim),
                    fovea_radius=fovea_radius,
                    image_out_dim=patch_dim,
                    ring_sigma_scaling_factor=1.0,
                ).astype(np.float32)
            ),
        )

        # Disable automatic optimization!
        # self.automatic_optimization = False

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        b, c, h, w = x.size()

        if self.do_add_pos_encoding:
            x_full = self._add_pos_encodings_to_img_batch(x)
        else:
            x_full = x

        patch_loss, patch_rec_loss, patch_kl_div = (
            0.0,
            0.0,
            0.0,
        )

        # patches: (b, patch_dim*patch_dim, c, h_out, w_out)
        patches = self._foveate_image(x_full)
        # n_patches = patches.size(3) * patches.size(4)
        # make grid of patches
        patch_vis = torchvision.utils.make_grid(patches[0].permute(2, 3, 1, 0).view(-1, 3, self.patch_dim, self.patch_dim), nrow=patches.size(3), pad_value=1).cpu()
        plt.imshow(patch_vis.permute(1, 2, 0) / 2 + 0.5)


        # def get_next_patch_from_pos(pos: torch.Tensor):
        #     # pos is a tensor of shape (b, 2)
        #     # get (patch_dim, patch_dim) patch from x_full given by pos as center,
        #     # where x and y are in [-1, 1]

        #     # calculate l2 distances from pos to positions of all pixels in x_full
        #     pos_l2_distances = torch.sqrt((x_full[:, -2:, :, :] - pos).pow(2).sum(dim=1))  # (b, h, w)
        #     # find the (x, y) pixel index in x_full that has the lowest distance
        #     min_distance_pos = pos_l2_distances.flatten(1, 2).argmin(dim=1)  # argmin over (b, h*w)
        #     # convert to (b, x, y) pixel index
        #     min_distance_pos = torch.stack(
        #         (min_distance_pos // w, min_distance_pos % w), dim=-1
        #     )
        #     # get the patch from x_full
        #     patch = x_full[:, :, min_distance_pos[:, 0] - self.patch_dim-1 , min_distance_pos[:, 1]]

        def get_next_patch(curr_patch: torch.Tensor, curr_z: torch.Tensor):
            curr_pos = curr_patch[:, -2:, :, :]

        for step in range(self.n_steps):
            patch = x_full[:, :, : self.patch_dim, : self.patch_dim]
            sample_zs, z_mus, z_logvars = self._encode_patch(patch)

            patch_recons = self._decode_patch(sample_zs[-1])
            assert torch.is_same_size(patch_recons[-1], patch)
            loss, rec_loss, kl_div = self._loss(
                patch, patch_recons[-1], sample_zs[0], z_mus[0], z_logvars[0]
            )
            patch_loss += loss
            patch_rec_loss += rec_loss
            patch_kl_div += kl_div

        return (patch_loss, patch_rec_loss, patch_kl_div), patch_recons[-1], z_mus[0], z_logvars[0]

    def _loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        z: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ):
        def gaussian_likelihood(x, x_hat, logscale):
            # scale = torch.exp(torch.ones_like(x_hat) * logscale)
            mean = x_hat
            dist = torch.distributions.Normal(mean, 1.0)

            # measure prob of seeing image under p(x|z)
            log_pxz = dist.log_prob(x)
            return log_pxz.sum(dim=(1, 2, 3))

        def kl_divergence(z, mu, std):
            # --------------------------
            # Monte carlo KL divergence
            # --------------------------
            # 1. define the first two probabilities (in this case Normal for both)
            p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
            q = torch.distributions.Normal(mu, std)

            # # 2. get the probabilities from the equation
            # # log(q(z|x)) - log(p(z))
            # log_qzx = q.log_prob(z)
            # log_pz = p.log_prob(z)
            # kl = (log_qzx - log_pz)
            kl = torch.distributions.kl_divergence(q, p)
            kl = kl.sum(-1)
            return kl

        std = torch.exp(logvar / 2)

        try:
            # can error due to bad predictions
            recon_loss = -gaussian_likelihood(x, x_recon, 0.0).mean()
        except ValueError as e:
            recon_loss = torch.nan

        try:
            kl = kl_divergence(z, mu, std).mean()
        except ValueError as e:
            kl = torch.nan

        # maximize reconstruction likelihood (minimize its negative), minimize kl divergence
        return (self.beta * kl + recon_loss), recon_loss, kl

    def _foveate_image(self, image: torch.Tensor):
        # image: (b, c, h, w)
        # filters: (out_h, out_w, rf_h, rf_w)
        # out: (b, c, out_h, out_w, rf_h, rf_w)
        # return torch.einsum("bchw,ijhw->bcij", image, self.foveation_filters)

        b, c, h, w = image.shape

        if self.foveation_padding_mode == "replicate":
            padding_mode = "replicate"
            pad_value = None
        elif self.foveation_padding_mode == "zeros":
            padding_mode = "constant"
            pad_value = 0.0
        else:
            raise ValueError(f"Unknown padding mode: {self.foveation_padding_mode}")

        if self.foveation_padding == "max":
            padded_image = F.pad(
                image,
                (h, h, w, w),
                mode=padding_mode,
                value=pad_value,
            )
        elif self.foveation_padding > 0:
            padded_image = F.pad(
                image,
                (
                    self.foveation_padding,
                    self.foveation_padding,
                    self.foveation_padding,
                    self.foveation_padding,
                ),
                mode=padding_mode,
                value=pad_value,
            )
        else:
            padded_image = image

        pad_h, pad_w = padded_image.shape[-2:]

        filter_rf_x, filter_rf_y = self.foveation_filters.shape[-2:]

        return F.conv3d(
            padded_image.view(b, 1, c, pad_h, pad_w),
            self.foveation_filters.view(-1, 1, 1, filter_rf_x, filter_rf_y),
            padding=0,
            stride=1,
        )

    def _encode_patch(self, x: torch.Tensor):
        mus, logvars, zs = [], [], []
        for encoder in self.encoders:
            z_mu, z_logvar = encoder(x)
            z = reparam_sample(z_mu, z_logvar)
            mus.append(z_mu)
            logvars.append(z_logvar)
            zs.append(z)
        return zs, mus, logvars

    def _decode_patch(self, z):
        decodings = []
        for decoder in self.decoders:
            dec = decoder(z)
            decodings.append(dec)

        decodings[-1] = decodings[-1].reshape(
            (-1, self.num_channels, self.patch_dim, self.patch_dim)
        )

        return decodings

    def _add_pos_encodings_to_img_batch(self, x: torch.Tensor):
        b, c, h, w = x.size()
        # add position encoding as in wattersSpatialBroadcastDecoder2019
        width_pos = torch.linspace(-1, 1, w)
        height_pos = torch.linspace(-1, 1, h)
        xb, yb = torch.meshgrid(width_pos, height_pos)
        # match dimensions of x except for channels
        xb = xb.expand(b, 1, -1, -1).to(x.device)
        yb = yb.expand(b, 1, -1, -1).to(x.device)
        x_full = torch.concat((x, xb, yb), dim=1)
        assert x_full.size() == torch.Size([b, c + 2, h, w])
        return x_full

    # generate n=num images using the model
    def generate(self, num: int):
        self.eval()
        z = torch.randn(num, self.z_dim)
        with torch.no_grad():
            return self._decode_patch(z)[-1].cpu()

    # returns z for position-augmented patch
    def get_patch_zs(self, patch_with_pos: torch.Tensor):
        assert patch_with_pos.ndim == 4
        self.eval()

        with torch.no_grad():
            zs, mus, logvars = self._encode_patch(patch_with_pos)

        return zs

    # def linear_interpolate(self, im1, im2):
    #     self.eval()
    #     z1 = self.get_z(im1)
    #     z2 = self.get_z(im2)

    #     factors = np.linspace(1, 0, num=10)
    #     result = []

    #     with torch.no_grad():
    #         for f in factors:
    #             z = f * z1 + (1 - f) * z2
    #             im = torch.squeeze(self._decode_patch(z).cpu())
    #             result.append(im)

    #     return result

    def latent_traverse(self, z, range_limit=3, step=0.5, around_z=False):
        self.eval()
        interpolation = torch.arange(-range_limit, range_limit + 0.1, step)
        samples = []
        with torch.no_grad():
            for row in range(self.z_dim):
                row_samples = []
                # copy to CPU to bypass https://github.com/pytorch/pytorch/issues/94390
                interp_z = z.clone().to("cpu")
                for val in interpolation:
                    if around_z:
                        interp_z[:, row] += val
                    else:
                        interp_z[:, row] = val
                    sample = self._decode_patch(interp_z.to(self.device))[-1].data.cpu()
                    row_samples.append(sample)
                samples.append(row_samples)
        return samples

    def training_step(self, batch, batch_idx):
        x, y = batch
        (loss, rec_loss, kl_div), x_recon, _, _ = self.forward(x, y)
        self.log("train_loss", loss)
        self.log("train_rec_loss", rec_loss, prog_bar=True)
        self.log("train_kl_div", kl_div, prog_bar=True)

        # self._optimizer_step(loss)
        skip_update = float(torch.isnan(loss))  # TODO: skip on grad norm
        if skip_update:
            print(f"Skipping update! {loss=}, {rec_loss=}, {kl_div=}")

        self.log(
            "n_skipped_steps",
            skip_update,
            on_epoch=True,
            on_step=False,
            logger=True,
            prog_bar=True,
            reduce_fx=torch.sum,
        )
        # self.log(grad_norm, skip_update, on_epoch=True, logger=True)

        return None if skip_update else loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        (loss, rec_loss, kl_div), x_recon, _, _ = self.forward(x, y)
        self.log("val_loss", loss)
        self.log("val_rec_loss", rec_loss)
        self.log("val_kl_div", kl_div)
        if batch_idx == 0:
            tensorboard = self.logger.experiment
            # real = torchvision.utils.make_grid(x).cpu()
            # recon = torchvision.utils.make_grid(x_recon).cpu()
            # img = torch.concat((real, recon), dim=1)

            # step constant bc real images don't change
            def remove_pos_channels_from_batch(g):
                n_pos_channels = 2 if self.do_add_pos_encoding else 0
                return g[:, :-n_pos_channels, :, :]

            tensorboard.add_images("Real Images", x[:32], global_step=0)
            tensorboard.add_images(
                "Reconstructed Images",
                remove_pos_channels_from_batch(x_recon[:32]),
                global_step=self.global_step,
            )

            # TODO: this traversal stuff makes no sense on sub-image patches!

            def stack_traversal_output(g):
                # stack by interp image, then squeeze out the singular batch dimension and index out the 2 position channels
                return [
                    remove_pos_channels_from_batch(torch.stack(dt).squeeze(1))
                    for dt in traversal_abs
                ]

            img = self._add_pos_encodings_to_img_batch(x[[0]])
            traversal_abs = self.latent_traverse(
                self.get_patch_zs(img)[-1], range_limit=3, step=0.5
            )
            images_by_row_and_interp = stack_traversal_output(traversal_abs)

            tensorboard.add_image(
                "Absolute Latent Traversal",
                torchvision.utils.make_grid(
                    torch.concat(images_by_row_and_interp), nrow=self.z_dim
                ),
                global_step=self.global_step,
            )
            traversal_around = self.latent_traverse(
                self.get_patch_zs(img)[-1], range_limit=3, step=0.5, around_z=True
            )
            images_by_row_and_interp = stack_traversal_output(traversal_around)

            tensorboard.add_image(
                "Latent Traversal Around Z",
                torchvision.utils.make_grid(
                    torch.concat(images_by_row_and_interp), nrow=self.z_dim
                ),
                global_step=self.global_step,
            )

        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    # def _get_grad_norm(self):
    #     total_norm = 0
    #     parameters = [p for p in self.parameters() if p.grad is not None and p.requires_grad]
    #     for p in parameters:
    #         param_norm = p.grad.detach().data.norm(2)
    #         total_norm += param_norm.item() ** 2
    #     total_norm = total_norm ** 0.5
    #     return total_norm

    # def _optimizer_step(self, loss):
    #     opt = self.optimizers()
    #     opt.zero_grad()
    #     self.manual_backward(loss)

    #     grad_norm = self.clip_gradients(opt, gradient_clip_val=self.grad_clip, gradient_clip_algorithm="norm")

    #     # only update if loss is not NaN and if the grad norm is below a specific threshold
    #     skipped_update = 1
    #     if not torch.isnan(loss) and (
    #         self.grad_skip_threshold == -1 or grad_norm < self.grad_skip_threshold
    #     ):
    #         skipped_update = 0
    #         opt.step()
    #         # TODO: EMA updating
    #         # TODO: factor out loss NaNs by what produced them (kl or reconstruction)
    #         # update_ema(vae, ema_vae, H.ema_rate)

    #     self.log("n_skipped_steps", skipped_update, on_epoch=True, logger=True)

    # def optimizer_step(
    #     self,
    #     epoch,
    #     batch_idx,
    #     optimizer,
    #     optimizer_idx,
    #     optimizer_closure,
    #     on_tpu=False,
    #     using_lbfgs=False,
    # ):
    #     # skip updates with nans
    #     if True:
    #         # the closure (which includes the `training_step`) will be executed by `optimizer.step`
    #         optimizer.step(closure=optimizer_closure)
    #     else:
    #         # call the closure by itself to run `training_step` + `backward` without an optimizer step
    #         optimizer_closure()


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()
