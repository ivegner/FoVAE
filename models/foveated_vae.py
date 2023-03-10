import gc
from copy import deepcopy
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
from torch import nn, optim

import utils.foveation as fov_utils
from modules.transformers import VisionTransformer
from utils.vae_utils import gaussian_kl_divergence, gaussian_likelihood
from utils.visualization import imshow_unnorm, plot_gaussian_foveation_parameters


def _recursive_to(x, *args, **kwargs):
    if isinstance(x, torch.Tensor):
        return x.to(*args, **kwargs)
    elif isinstance(x, dict):
        return {k: _recursive_to(v, *args, **kwargs) for k, v in x.items()}
    elif isinstance(x, list):
        return [_recursive_to(v, *args, **kwargs) for v in x]
    else:
        return x


def reparam_sample(mu, logvar):
    std = torch.exp(0.5 * logvar)  # e^(1/2 * log(std^2))
    i = 0
    while i < 5:
        # randn_like sometimes produces NaNs for unknown reasons
        # maybe see: https://github.com/pytorch/pytorch/issues/46155
        # so we try again if that happens
        eps = torch.randn_like(std)  # random ~ N(0, 1)
        if not torch.isnan(eps).any():
            break
        i += 1
    else:
        raise RuntimeError("Could not sample from N(0, 1) without NaNs after 5 tries")
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
        patch_dim=6,
        patch_channels=3,
        # n_vae_levels=1,
        z_dim=10,
        n_steps: int = 1,
        foveation_padding: Union[Literal["max"], int] = "max",
        foveation_padding_mode: Literal["zeros", "replicate"] = "replicate",
        lr=1e-3,
        beta=1,
        # grad_clip=100,
        grad_skip_threshold=1000,
        # do_add_pos_encoding=True,
        do_z_pred_cond_from_top=True,
        do_use_beta_norm=True,
        do_random_foveation=False,
        do_image_reconstruction=True,
        do_next_patch_prediction=True,
    ):
        super().__init__()

        self.image_dim = image_dim
        self.fovea_radius = fovea_radius
        self.patch_dim = patch_dim
        self.z_dim = z_dim

        # self.n_vae_levels = n_vae_levels

        self.n_steps = n_steps
        # if do_add_pos_encoding:
        self.num_channels = patch_channels + 2
        self.lr = lr
        self.foveation_padding = foveation_padding
        self.foveation_padding_mode = foveation_padding_mode

        input_dim = self.patch_dim * self.patch_dim * self.num_channels

        self.encoders = nn.ModuleList([UpBlock(input_dim, z_dim, [256, 256])])
        self.decoders = nn.ModuleList([DownBlock(z_dim, input_dim, [256, 256])])

        # for all levels except the highest-level next-patch-z predictor, the predictors are
        # DownBlocks analogous to the decoders
        # for highest-level next-patch-z predictor, the predictor is a Transformer conditioned on
        # all past top-level Zs
        self.next_patch_predictors = nn.ModuleList(
            [
                VisionTransformer(
                    input_dim=z_dim + 2,  # 2 for concatenated next position
                    output_dim=z_dim,
                    embed_dim=32,  # TODO
                    hidden_dim=64,  # TODO
                    num_heads=1,  # TODO
                    num_layers=1,  # TODO
                    dropout=0,
                ),
                DownBlock(z_dim * 2 if do_z_pred_cond_from_top else z_dim, input_dim),
            ]
        )

        self.next_location_predictor = VisionTransformer(
            input_dim=z_dim,
            output_dim=2 * 2,
            embed_dim=32,  # TODO
            hidden_dim=64,  # TODO
            num_heads=1,  # TODO
            num_layers=1,  # TODO
            dropout=0,
        )

        self._beta = beta

        self.grad_skip_threshold = grad_skip_threshold
        # self.do_add_pos_encoding = do_add_pos_encoding
        self.do_z_pred_cond_from_top = do_z_pred_cond_from_top

        if do_use_beta_norm:
            beta_vae = (beta * z_dim) / input_dim  # according to beta-vae paper
            print(
                f"Using normalized betas[1] value of {beta_vae:.6f} as beta, calculated from unnormalized beta_vae {beta:.6f}"
            )
        else:
            beta_vae = beta

        self.betas = dict(
            curr_patch_recon=1,
            curr_patch_kl=beta_vae,
            next_patch_recon=100,
            next_patch_pos_kl=10,
            image_recon=100,
        )

        # image: (b, c, image_dim[0], image_dim[1])
        # TODO: sparsify
        self.default_gaussian_filter_params = (
            fov_utils.get_default_gaussian_foveation_filter_params(
                image_dim=(image_dim, image_dim),
                fovea_radius=fovea_radius,
                image_out_dim=patch_dim,
                ring_sigma_scaling_factor=2,  # in pyramidal case, pixel ring i averages 2^i pixels
            )
        )
        self.do_random_foveation = do_random_foveation
        self.do_image_reconstruction = do_image_reconstruction
        self.do_next_patch_prediction = do_next_patch_prediction

        # Disable automatic optimization!
        # self.automatic_optimization = False

    def to(self, *args, **kwargs):
        g = super().to(*args, **kwargs)
        g.default_gaussian_filter_params = _recursive_to(
            self.default_gaussian_filter_params, *args, **kwargs
        )
        return g

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        b, c, h, w = x.size()

        # if self.do_add_pos_encoding:
        x_full = self._add_pos_encodings_to_img_batch(x)
        # else:
        #     x_full = x

        curr_patch_total_loss, curr_patch_rec_total_loss, curr_patch_total_kl_div = (
            0.0,
            0.0,
            0.0,
        )
        next_patch_z_pred_total_loss = 0.0
        next_patch_pos_kl_div = 0.0
        # next_patch_total_loss, next_patch_rec_loss, next_patch_kl_div = (
        #     0.0,
        #     0.0,
        #     0.0,
        # )

        # patches: (b, patch_dim*patch_dim, c, h_out, w_out)
        # patch_locs_x = torch.linspace(-1, 1, 30)
        # patch_locs_y = torch.linspace(-1, 1, 30)
        # patches = [self._foveate_to_loc(x_full, loc)[0] for loc in torch.cartesian_prod(patch_locs_x, patch_locs_y)]
        # patch_vis = torchvision.utils.make_grid(patches, nrow=len(patch_locs_x), pad_value=1).cpu()
        # plt.imshow(patch_vis.permute(1, 2, 0) / 2 + 0.5)

        # def get_next_patch(curr_patch: torch.Tensor, curr_zs: torch.Tensor):
        #     curr_pos = curr_patch.view(b, self.num_channels, self.patch_dim, self.patch_dim)[
        #         :, -2:, :, :
        #     ].mean(dim=(2, 3))
        #     # move pos in random direction a small amount
        #     next_pos = curr_pos + torch.randn_like(curr_pos) * 0.1
        #     # clamp to [-1, 1]
        #     next_pos = torch.clamp(next_pos, -1, 1)
        #     # get next patch
        #     next_patch = get_patch_from_pos(next_pos)
        #     return next_patch, next_pos

        def get_patch_from_pos(pos):
            # TODO: investigate why reshape vs. view is needed
            patch = self._foveate_to_loc(x_full, pos).reshape(b, -1)
            assert patch.shape == (b, self.num_channels * self.patch_dim * self.patch_dim)
            return patch

        initial_positions = torch.tensor([0.0, 0.0], device=x.device).unsqueeze(0).repeat(b, 1)
        patches = [get_patch_from_pos(initial_positions)]
        sample_zs = []
        pred_sample_zs = []
        z_mus = []
        z_logvars = []
        z_recons = []
        patch_positions = [initial_positions]
        for step in range(self.n_steps):
            curr_patch = patches[-1]
            curr_sample_zs, curr_z_mus, curr_z_logvars = self._encode_patch(curr_patch)

            # if any previous predicted patch, calculate loss between current patch and previous predicted patch
            if self.do_next_patch_prediction and len(pred_sample_zs) > 0:
                prev_pred_sample_zs = pred_sample_zs[-1]
                for i in range(len(prev_pred_sample_zs)):
                    next_patch_z_pred_total_loss += F.mse_loss(
                        prev_pred_sample_zs[i], curr_sample_zs[i]
                    )

            # reconstruct current patch
            curr_z_recons = self._decode_patch(curr_sample_zs[-1])  # decode from top-level z
            curr_patch_recon = curr_z_recons[-1]
            assert torch.is_same_size(curr_patch_recon, curr_patch)

            # TODO: loss on all levels
            _curr_patch_rec_loss, _curr_patch_kl_div = self._loss(
                curr_patch, curr_patch_recon, curr_z_mus[-1], curr_z_logvars[-1]
            )
            # curr_patch_total_loss += _curr_patch_total_loss
            curr_patch_rec_total_loss += _curr_patch_rec_loss
            curr_patch_total_kl_div += _curr_patch_kl_div

            sample_zs.append(curr_sample_zs)
            z_mus.append(curr_z_mus)
            z_logvars.append(curr_z_logvars)
            z_recons.append(curr_z_recons)

            (
                next_patch_sample_pos,
                next_patch_pos_mus,
                next_patch_pos_logvars,
            ) = self._predict_next_patch_loc(sample_zs)
            # calculate kl divergence between predicted next patch pos and std-normal prior
            # only do kl divergence because reconstruction of next_pos is captured in next_patch_rec_loss
            if not self.do_random_foveation:
                _, _next_patch_pos_kl_div = self._loss(
                    mu=next_patch_pos_mus, logvar=next_patch_pos_logvars
                )
                next_patch_pos_kl_div += _next_patch_pos_kl_div
            patch_positions.append(next_patch_sample_pos)

            # predict zs for next patch
            pred_next_zs = self._predict_next_patch_zs(sample_zs, next_patch_sample_pos)
            pred_sample_zs.append(pred_next_zs)
            pred_next_patch = pred_next_zs[0]  # analogous to curr_sample_zs[0]
            pred_next_patch_pos = pred_next_patch.view(
                b, self.num_channels, self.patch_dim, self.patch_dim
            )[:, -2:, :, :].mean(dim=(2, 3))
            # clamp to [-1, 1]
            pred_next_patch_pos = torch.clamp(pred_next_patch_pos, -1, 1)

            # foveate to next position
            next_patch = get_patch_from_pos(next_patch_sample_pos)
            assert torch.is_same_size(next_patch, curr_patch)

            # check that the next patch's position is close to the position from which
            # it was supposed to be extracted
            # position will wobble a little due to gaussian aggregation? TODO: investigate
            # I don't care as long as it's under half a pixel

            # this is commented because we do not allow foveation outside the image, and
            # depending on method of padding, locations in the padding area will not have true locations, but
            # will be clamped to locations at the edge of the image (or zeros). As such, averaging the locations on the patch
            # has to be inside the image, but the location would encompass locations outside the image.
            # this is a useful check though, and should be re-enabled if the foveation method is changed to not rely on
            # padding and/or clamping
            # _next_patch_center = next_patch.view(
            #     b, self.num_channels, self.patch_dim, self.patch_dim
            # )[:, -2:, :, :].mean(dim=(2, 3))
            # _acceptable_dist_threshold = 0.5 / min(h, w)
            # assert (
            #     _next_patch_center - next_patch_sample_pos
            # ).abs().max() <= _acceptable_dist_threshold, (
            #     f"Next patch position {_next_patch_center.round(2).cpu()} is too far from predicted position {next_patch_sample_pos.round(2).cpu()}: "
            #     f"{(_next_patch_center - next_patch_sample_pos).abs().max()} > {_acceptable_dist_threshold}"
            # )

            patches.append(next_patch)

        if self.do_image_reconstruction:
            image_reconstruction_loss, _ = self._reconstruct_image(
                sample_zs, x_full, return_patches=False
            )
        else:
            image_reconstruction_loss = torch.tensor(0.0, device=self.device)

        # TODO: there's a memory leak somewhere, comes out during overfit_batches=1
        # Notes on memory leak:
        # https://github.com/Lightning-AI/lightning/issues/16876
        # https://github.com/pytorch/pytorch/issues/13246
        # https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/

        curr_patch_rec_total_loss = (
            curr_patch_rec_total_loss * self.betas["curr_patch_recon"] / self.n_steps
        )
        curr_patch_total_kl_div = (
            curr_patch_total_kl_div * self.betas["curr_patch_kl"] / self.n_steps
        )
        next_patch_z_pred_total_loss = (
            next_patch_z_pred_total_loss * self.betas["next_patch_recon"] / self.n_steps
        )
        next_patch_pos_kl_div = (
            next_patch_pos_kl_div * self.betas["next_patch_pos_kl"] / self.n_steps
        )
        image_reconstruction_loss = image_reconstruction_loss * self.betas["image_recon"]

        # curr_patch_total_loss = curr_patch_rec_total_loss + self.beta * curr_patch_total_kl_div

        total_loss = (
            curr_patch_rec_total_loss
            + curr_patch_total_kl_div
            + next_patch_z_pred_total_loss
            + next_patch_pos_kl_div
            + image_reconstruction_loss
        )

        return dict(
            losses=dict(
                total_loss=total_loss,
                curr_patch_total_loss=curr_patch_rec_total_loss + curr_patch_total_kl_div,
                curr_patch_rec_loss=curr_patch_rec_total_loss,
                curr_patch_kl_div=curr_patch_total_kl_div,
                next_patch_z_pred_total_loss=next_patch_z_pred_total_loss,
                next_patch_pos_kl_div=next_patch_pos_kl_div,
                image_reconstruction_loss=image_reconstruction_loss,
            ),
            step_vars=dict(
                # all except last are a list of length n_steps, each element is a list of length n_levels
                sample_zs=sample_zs,
                curr_z_mus=curr_z_mus,
                curr_z_logvars=curr_z_logvars,
                z_recons=z_recons,
                pred_sample_zs=pred_sample_zs,
                patch_positions=patch_positions,  # list of length n_steps, each element is a tensor of shape (b, 2)
                # image_reconstruction_patches=image_reconstruction_patches,  # (b, n_patches, n_channels, patch_dim, patch_dim)
            ),
        )

    def _reconstruct_image(self, sample_zs, image: Optional[torch.Tensor], return_patches=False):
        # positions span [-1, 1] in both x and y

        b = sample_zs[0][0].size(0)

        positions_x = torch.linspace(
            -1, 1, steps=int(np.ceil(self.image_dim / (self.fovea_radius * 2))), device=self.device
        )
        positions_y = torch.linspace(
            -1, 1, steps=int(np.ceil(self.image_dim / (self.fovea_radius * 2))), device=self.device
        )
        positions = torch.stack(torch.meshgrid(positions_x, positions_y, indexing="xy"), dim=-1)
        positions = positions.view(-1, 2).unsqueeze(1).expand(-1, b, -1)

        # predict zs for each position
        image_recon_loss = None
        pred_zs = [*sample_zs]
        patches = []
        # TODO: maybe reconstruct only some patches?
        for position in positions:
            pred_step_zs = self._predict_next_patch_zs(pred_zs, position)
            # pred_zs.append(pred_step_zs)
            pred_patch = pred_step_zs[0]
            pred_patch = pred_patch.view(b, self.num_channels, self.patch_dim, self.patch_dim)
            if image is not None:
                if image_recon_loss is None:
                    image_recon_loss = 0.0
                real_patch = self._foveate_to_loc(image, position)  # SLOWWWWW
                assert torch.is_same_size(pred_patch, real_patch)
                # TODO: mask to fovea only
                patch_recon_loss = F.mse_loss(pred_patch, real_patch)
                image_recon_loss += patch_recon_loss
            patches.append(pred_patch)

        if image_recon_loss is not None:
            image_recon_loss /= positions.size(0)

        if return_patches:
            return image_recon_loss, torch.stack(patches, dim=0).transpose(0, 1)
        else:
            return image_recon_loss, None

    def _loss(
        self,
        x: Optional[torch.Tensor] = None,
        x_recon: Optional[torch.Tensor] = None,
        # z: Optional[torch.Tensor] = None,
        mu: Optional[torch.Tensor] = None,
        logvar: Optional[torch.Tensor] = None,
    ):
        """
        Calculate Gaussian likelihood + KL divergence for an arbitrary input.
        If x and x_recon are provided, Gaussian likelihood loss is calculated.
        If mu and logvar are provided, KL divergence loss from standard-normal prior is calculated.
        If both are provided, both losses are calculated and summed weighted by self.beta.

        Args:
            x: input
            x_recon: reconstruction of input
            mu: latent mean
            logvar: latent log variance

        Returns:
            loss: beta-weighted sum of Gaussian likelihood and KL divergence losses, if both are calculated
            recon_loss: Gaussian likelihood loss, if calculated
            kl: KL divergence loss, if calculated
        """
        recon_loss, kl = None, None

        if (x is None and x_recon is None) and (mu is None and logvar is None):
            raise ValueError(
                "Must provide either x and x_recon (for likelihood loss) or mu and logvar (for KL divergence loss)"
            )

        if (x is not None and x_recon is None) or (x is None and x_recon is not None):
            raise ValueError("Must provide both x and x_recon for likelihood loss")
        else:
            try:
                # can error due to bad predictions
                recon_loss = -gaussian_likelihood(x, x_recon, 0.0).mean()
            except ValueError as e:
                recon_loss = torch.nan

        if (mu is not None and logvar is None) or (mu is None and logvar is not None):
            raise ValueError("Must provide both mu and logvar for KL divergence loss")
        else:
            try:
                std = torch.exp(logvar / 2)
                kl = gaussian_kl_divergence(mu, std).mean()
            except ValueError as e:
                kl = torch.nan

        # total_loss = (
        #     (self.beta * kl + recon_loss) if recon_loss is not None and kl is not None else None
        # )

        # maximize reconstruction likelihood (minimize its negative), minimize kl divergence
        return recon_loss, kl

    def _foveate_to_loc(self, image: torch.Tensor, loc: torch.Tensor):
        # image: (b, c, h, w)
        # loc: (b, 2), where entries are in [-1, 1]
        # filters: (out_h, out_w, rf_h, rf_w)
        # out: (b, c, out_h, out_w, rf_h, rf_w)

        b, c, h, w = image.shape

        if self.foveation_padding_mode == "replicate":
            padding_mode = "replicate"
            pad_value = None
        elif self.foveation_padding_mode == "zeros":
            padding_mode = "constant"
            pad_value = 0.0
        else:
            raise ValueError(f"Unknown padding mode: {self.foveation_padding_mode}")

        pad_offset = [0, 0]
        if self.foveation_padding == "max":
            pad_offset = np.ceil([h / 2, w / 2]).astype(int).tolist()
            padded_image = F.pad(
                image,
                (pad_offset[0], pad_offset[0], pad_offset[1], pad_offset[1]),
                mode=padding_mode,
                value=pad_value,
            )
        elif self.foveation_padding > 0:
            pad_offset = [self.foveation_padding, self.foveation_padding]
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

        # pad_h, pad_w = padded_image.shape[-2:]

        # move the gaussian filter params to the loc
        gaussian_filter_params = self._move_default_filter_params_to_loc(loc, (h, w), pad_offset)

        # foveate
        foveated_image = fov_utils.apply_mean_foveation_pyramid(
            padded_image, gaussian_filter_params
        )
        if foveated_image.isnan().any():
            raise ValueError("NaNs in foveated image!")
        return foveated_image

    def _move_default_filter_params_to_loc(
        self, loc: torch.Tensor, image_dim: Iterable, pad_offset: Optional[Iterable] = None
    ):
        """Move gaussian foveation params centers to center at loc in the padded image"""
        assert len(image_dim) == 2, f"image_dim must be (h, w), got {image_dim}"
        image_dim = torch.tensor(image_dim, dtype=loc.dtype, device=loc.device)

        assert (
            -1 <= loc.min() and loc.max() <= 1
        ), f"loc must be in [-1, 1], got {loc.min()} to {loc.max()}"

        if pad_offset is None:
            pad_offset = [0, 0]
        else:
            assert len(pad_offset) == 2, f"pad_offset must be (h, w), got {pad_offset}"

        # convert loc in [-1, 1] to (b, x, y) pixel index in the original image
        loc = (loc + 1) / 2
        loc = loc * image_dim

        generic_center = image_dim / 2
        pad_offset = torch.tensor(pad_offset, dtype=loc.dtype, device=loc.device).unsqueeze(0)

        gaussian_filter_params = deepcopy(self.default_gaussian_filter_params)  # TODO: optimize
        for ring in [gaussian_filter_params["fovea"], *gaussian_filter_params["peripheral_rings"]]:
            new_mus = ring["mus"] + (loc - generic_center).unsqueeze(1)
            if not torch.isclose(new_mus.mean(1), loc, atol=1e-4).all():
                print(
                    f"New gaussian centers after move not close to loc: {new_mus.mean(1)[torch.argmax((new_mus.mean(1) - loc).sum(1), 0)]} vs {loc[torch.argmax((new_mus.mean(1) - loc).sum(1), 0)]}"
                )
            ring["mus"] = new_mus + pad_offset

        return gaussian_filter_params

    def _encode_patch(self, x: torch.Tensor):
        mus, logvars, zs = [None], [None], [x]
        for encoder in self.encoders:
            z_mu, z_logvar = encoder(zs[-1])
            z = reparam_sample(z_mu, z_logvar)
            mus.append(z_mu)
            logvars.append(z_logvar)
            zs.append(z)
        return zs, mus, logvars

    def _decode_patch(self, z, z_level=None):
        decodings = [z]

        if z_level is not None:
            decoders = self.decoders[-z_level:]
        else:
            decoders = self.decoders

        # iterate decoders from highest level to lowest
        for decoder in decoders:
            dec = decoder(decodings[-1])
            decodings.append(dec)

        # decodings[-1] = decodings[-1].reshape(
        #     (-1, self.num_channels, self.patch_dim, self.patch_dim)
        # )

        return decodings

    def _predict_next_patch_loc(self, prev_zs: List[List[torch.Tensor]]):
        # prev_zs: list(n_steps_so_far) of lists (n_levels from lowest to highest) of tensors (b, dim)
        # highest-level z is the last element of the list

        Z_LEVEL_TO_PRED_LOC = -1  # TODO: make this a param, and maybe multiple levels

        if self.do_random_foveation:
            return torch.rand((prev_zs[0][0].size(0), 2), device=prev_zs[0][0].device) * 2 - 1, None, None

        else:
            prev_top_zs = torch.stack([zs[Z_LEVEL_TO_PRED_LOC] for zs in prev_zs], dim=0)
            prev_top_zs = prev_top_zs.transpose(0, 1)  # (b, n_steps, dim)
            pred = self.next_location_predictor(prev_top_zs)
            next_loc_mu, next_loc_logvar = pred[:, :2], pred[:, 2:]
            next_loc = reparam_sample(next_loc_mu, next_loc_logvar)

            return (
                # F.sigmoid(next_loc)*2 - 1,
                torch.clamp(next_loc, -1, 1),
                next_loc_mu,
                next_loc_logvar,
            )

    def _predict_next_patch_zs(
        self, prev_zs: List[List[torch.Tensor]], next_patch_pos: torch.Tensor
    ):
        # prev_zs: list(n_steps_so_far) of lists (n_levels from lowest to highest) of tensors (b, dim)
        # next_patch_pos: Tensor (b, 2)
        # highest-level z is the last element of the list

        # TODO: should these be sampled (i.e. predict + sample from mu+logvar) instead of predicted?
        # I don't think so because trying to be analogous to the decoder, which doesn't sample

        next_zs = []

        N_LEVELS = 1 + 1
        for i, predictor in enumerate(self.next_patch_predictors):
            z_level_to_predict = N_LEVELS - i - 1
            if z_level_to_predict == N_LEVELS - 1:
                # predict next highest-level z using special Transformer procedure
                # concat all previous top-level zs

                # # add next patch pos as extra token
                # # TODO: maybe add concatenate it to all tokens instead?
                # pos_as_token = torch.zeros_like(prev_zs[0][-1])
                # pos_as_token[:, :2] = next_patch_pos

                prev_top_zs = torch.stack([zs[-1] for zs in prev_zs], dim=0)
                prev_top_zs = prev_top_zs.transpose(0, 1)  # (b, n_steps, dim)

                # concatenate next patch pos to each z, TODO: maybe add as extra token instead?
                prev_top_zs_with_pos = torch.cat(
                    (prev_top_zs, next_patch_pos.unsqueeze(1).repeat(1, prev_top_zs.size(1), 1)),
                    dim=2,
                )

                pred_z = self.next_patch_predictors[0](prev_top_zs_with_pos)
            else:
                # predict next z using previous z

                # TODO: condition on previous z of the same level?
                # prev_same_level_z = prev_zs[-1][z_level_to_predict]
                prev_higher_level_z = prev_zs[-1][z_level_to_predict + 1]
                x = prev_higher_level_z
                # condition on current predicted z of higher level?
                if self.do_z_pred_cond_from_top:
                    curr_higher_level_pred_z = next_zs[0]  # 0 because prepended
                    x = torch.cat((x, curr_higher_level_pred_z), dim=1)

                pred_z = predictor(x)

            next_zs = [pred_z] + next_zs  # prepend to match order of prev_zs

        return next_zs

    def _add_pos_encodings_to_img_batch(self, x: torch.Tensor):
        b, c, h, w = x.size()
        # add position encoding as in wattersSpatialBroadcastDecoder2019
        height_pos = torch.linspace(-1, 1, h)
        width_pos = torch.linspace(-1, 1, w)
        xb, yb = torch.meshgrid(height_pos, width_pos, indexing="xy")
        # match dimensions of x except for channels
        xb = xb.expand(b, 1, -1, -1).to(x.device)
        yb = yb.expand(b, 1, -1, -1).to(x.device)
        x_full = torch.concat((x, xb, yb), dim=1)
        assert x_full.size() == torch.Size([b, c + 2, h, w])
        return x_full

    # # generate n=num images using the model
    # def generate(self, num: int):
    #     self.eval()
    #     z = torch.randn(num, self.z_dim)
    #     with torch.no_grad():
    #         return self._decode_patch(z)[-1].cpu()

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

    def latent_traverse(self, z, z_level=1, range_limit=3, step=0.5, around_z=False):
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
                    sample = (
                        self._decode_patch(interp_z.to(self.device), z_level=z_level)[-1]
                        .data.cpu()
                        .reshape(z.shape[0], self.num_channels, self.patch_dim, self.patch_dim)
                    )
                    row_samples.append(sample)
                samples.append(row_samples)
        return samples

    def training_step(self, batch, batch_idx):
        x, y = batch
        forward_out = self.forward(x, y)
        total_loss = forward_out["losses"]["total_loss"]
        curr_patch_total_loss = forward_out["losses"]["curr_patch_total_loss"]
        curr_patch_rec_loss = forward_out["losses"]["curr_patch_rec_loss"]
        curr_patch_kl_div = forward_out["losses"]["curr_patch_kl_div"]
        next_patch_z_pred_total_loss = forward_out["losses"]["next_patch_z_pred_total_loss"]
        next_patch_pos_kl_div = forward_out["losses"]["next_patch_pos_kl_div"]
        image_reconstruction_loss = forward_out["losses"]["image_reconstruction_loss"]

        self.log("train_total_loss", total_loss)
        self.log("train_curr_patch_total_loss", curr_patch_total_loss)
        self.log("train_next_patch_zpred_total_loss", next_patch_z_pred_total_loss)
        self.log("train_curr_patch_rec_loss", curr_patch_rec_loss, prog_bar=True)
        self.log("train_curr_patch_kl_div", curr_patch_kl_div, prog_bar=True)
        self.log("train_next_patch_pos_kl_div", next_patch_pos_kl_div)
        self.log("train_image_recon_loss", image_reconstruction_loss)

        # self._optimizer_step(loss)
        skip_update = float(torch.isnan(total_loss))  # TODO: skip on grad norm
        if skip_update:
            print(f"Skipping update!", forward_out["losses"])

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

        return None if skip_update else total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        forward_out = self.forward(x, y)
        total_loss = forward_out["losses"]["total_loss"]
        curr_patch_total_loss = forward_out["losses"]["curr_patch_total_loss"]
        curr_patch_rec_loss = forward_out["losses"]["curr_patch_rec_loss"]
        curr_patch_kl_div = forward_out["losses"]["curr_patch_kl_div"]
        next_patch_z_pred_total_loss = forward_out["losses"]["next_patch_z_pred_total_loss"]
        next_patch_pos_kl_div = forward_out["losses"]["next_patch_pos_kl_div"]
        image_reconstruction_loss = forward_out["losses"]["image_reconstruction_loss"]

        step_sample_zs = forward_out["step_vars"]["sample_zs"]
        step_z_recons = forward_out["step_vars"]["z_recons"]
        step_next_z_preds = forward_out["step_vars"]["pred_sample_zs"]
        step_patch_positions = forward_out["step_vars"]["patch_positions"]

        # step_sample_zs: (n_steps, n_layers, batch_size, z_dim)
        # step_z_recons: (n_steps, n_layers, batch_size, z_dim)
        # last step_z_recons is the one used for reconstruction: step_z_recons[0, 0, -1].size()=n_channels*patch_dim*patch_dim
        assert (
            step_z_recons[0][-1][0].size()
            == step_sample_zs[0][0][0].size()
            == torch.Size([self.num_channels * self.patch_dim * self.patch_dim])
        )
        assert (
            step_sample_zs[0][1][0].size()
            == step_z_recons[0][0][-2].size()
            == torch.Size([self.z_dim])
        )
        self.log("val_total_loss", total_loss)
        self.log("val_curr_patch_total_loss", curr_patch_total_loss)
        self.log("val_next_patch_zpred_total_loss", next_patch_z_pred_total_loss)
        self.log("val_curr_patch_rec_loss", curr_patch_rec_loss)
        self.log("val_curr_patch_kl_div", curr_patch_kl_div)
        self.log("val_next_patch_pos_kl_div", next_patch_pos_kl_div)
        self.log("val_image_recon_loss", image_reconstruction_loss)

        if batch_idx == 0:
            N_TO_PLOT = 4
            tensorboard = self.logger.experiment
            # real = torchvision.utils.make_grid(x).cpu()
            # recon = torchvision.utils.make_grid(x_recon).cpu()
            # img = torch.concat((real, recon), dim=1)

            def remove_pos_channels_from_batch(g):
                n_pos_channels = 2  # if self.do_add_pos_encoding else 0
                return g[:, :-n_pos_channels, :, :]

            real_images = x[:N_TO_PLOT].repeat(self.n_steps, 1, 1, 1, 1)
            # plot stepwise foveations on real images
            h, w = real_images.shape[3:]

            # # # # DEBUG: demo foveation to a specific location
            # fig, (ax1, ax2) = plt.subplots(2)
            # loc = torch.tensor([0.0, 0.0]).repeat(1, 1).to("mps")
            # gaussian_filter_params = _recursive_to(self._move_default_filter_params_to_loc(loc, (h, w), pad_offset=None), "cpu",)
            # plot_gaussian_foveation_parameters(
            #                     x[[3]].cpu(),
            #                     gaussian_filter_params,
            #                     axs=[ax1],
            #                     point_size=10,
            #                 )
            # fov = self._foveate_to_loc(self._add_pos_encodings_to_img_batch(x[[3]]), loc).cpu()
            # imshow_unnorm(fov[0,[0]], ax=ax2)

            # make figure with a column for each step and 3 rows: 1 for image with foveation, one for patch, one for patch reconstruction

            figs = [plt.figure(figsize=(self.n_steps * 3, 12)) for _ in range(N_TO_PLOT)]
            axs = [f.subplots(4, self.n_steps) for f in figs]

            # plot foveations on images
            for step, img_step_batch in enumerate(real_images):
                # positions = (
                #     step_sample_zs[step][0]
                #     .view(-1, self.num_channels, self.patch_dim, self.patch_dim)[:N_TO_PLOT, -2:]
                #     .mean(dim=(2, 3))
                # )
                positions = step_patch_positions[step]
                gaussian_filter_params = _recursive_to(
                    self._move_default_filter_params_to_loc(positions, (h, w), pad_offset=None),
                    "cpu",
                )
                plot_gaussian_foveation_parameters(
                    img_step_batch.cpu(),
                    gaussian_filter_params,
                    axs=[a[0][step] for a in axs],
                    point_size=10,
                )
                for ax in [a[0][step] for a in axs]:
                    ax.set_title(f"Foveation at step {step}", fontsize=8)

            # plot patches
            for step in range(self.n_steps):
                patches = remove_pos_channels_from_batch(
                    step_sample_zs[step][0][:N_TO_PLOT].view(
                        -1, self.num_channels, self.patch_dim, self.patch_dim
                    )
                )
                for i in range(N_TO_PLOT):
                    imshow_unnorm(patches[i].cpu(), ax=axs[i][1][step])
                    axs[i][1][step].set_title(f"Patch at step {step}", fontsize=8)

            # plot patch reconstructions
            for step in range(self.n_steps):
                patches = remove_pos_channels_from_batch(
                    step_z_recons[step][-1][:N_TO_PLOT].view(
                        -1, self.num_channels, self.patch_dim, self.patch_dim
                    )
                )
                for i in range(N_TO_PLOT):
                    imshow_unnorm(patches[i].cpu(), ax=axs[i][2][step])
                    axs[i][2][step].set_title(f"Patch reconstruction at step {step}", fontsize=8)

            # plot next patch predictions
            for step in range(self.n_steps):
                pred_patches = step_next_z_preds[step][0][:N_TO_PLOT].view(
                    -1, self.num_channels, self.patch_dim, self.patch_dim
                )
                pred_pos = (pred_patches[:, -2:].mean(dim=(2, 3)) / 2 + 0.5).cpu() * torch.tensor(
                    [h, w]
                )
                pred_patches = remove_pos_channels_from_batch(pred_patches)
                for i in range(N_TO_PLOT):
                    ax = axs[i][3][step]
                    imshow_unnorm(pred_patches[i].cpu(), ax=ax)
                    ax.set_title(
                        f"Next patch pred. at step {step} - ({pred_pos[i][0]:.1f}, {pred_pos[i][1]:.1f})",
                        fontsize=8,
                    )
                    # ax.text(
                    #     -0.05,
                    #     -0.05,
                    #     f"(pred: {pred_pos[i][0]:.2f}, {pred_pos[i][1]:.2f})",
                    #     color="white",
                    #     fontsize=8,
                    #     bbox=dict(facecolor="black", alpha=0.5),
                    #     horizontalalignment="left",
                    #     verticalalignment="top",
                    # )

            # add to tensorboard
            for i, fig in enumerate(figs):
                fig.tight_layout()
                tensorboard.add_figure(f"Foveation Vis {i}", figs[i], global_step=self.global_step)
                del fig

            plt.close("all")

            if self.do_image_reconstruction:
                _, reconstructed_images = self._reconstruct_image(
                    [[level[:N_TO_PLOT] for level in step] for step in step_sample_zs],
                    image=None,
                    return_patches=True,
                )

                for i in range(N_TO_PLOT):
                    # ax = axs[i]
                    # imshow_unnorm(patches[i].cpu(), ax=ax)
                    # ax.set_title(
                    #     f"Next patch pred. at step {step} - ({pred_pos[i][0]:.1f}, {pred_pos[i][1]:.1f})",
                    #     fontsize=8,
                    # )
                    tensorboard.add_image(
                        f"Image Reconstructions {i}",
                        torchvision.utils.make_grid(
                            remove_pos_channels_from_batch(reconstructed_images[i]) / 2 + 0.5,
                            nrow=int(np.sqrt(len(reconstructed_images[i]))),
                        ),
                        global_step=self.global_step,
                    )

            # step constant bc real images don't change
            tensorboard.add_images(
                "Real Patches",
                remove_pos_channels_from_batch(
                    step_sample_zs[0][0][:32].view(
                        -1, self.num_channels, self.patch_dim, self.patch_dim
                    )
                    / 2
                    + 0.5
                ),
                global_step=0,
            )
            tensorboard.add_images(
                "Reconstructed Patches",
                remove_pos_channels_from_batch(
                    step_z_recons[0][-1][:32].view(
                        -1, self.num_channels, self.patch_dim, self.patch_dim
                    )
                    / 2
                    + 0.5
                ),
                global_step=self.global_step,
            )

            def stack_traversal_output(g):
                # stack by interp image, then squeeze out the singular batch dimension and index out the 2 position channels
                return [
                    remove_pos_channels_from_batch(torch.stack(dt).squeeze(1))
                    for dt in traversal_abs
                ]

            # img = self._add_pos_encodings_to_img_batch(x[[0]])
            # get first-level z of first step of first image of batch.
            z_level = 1
            first_step_zs = step_sample_zs[0][z_level][0].unsqueeze(0)
            traversal_abs = self.latent_traverse(
                first_step_zs, z_level=z_level, range_limit=3, step=0.5
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
                first_step_zs, z_level=z_level, range_limit=3, step=0.5, around_z=True
            )
            images_by_row_and_interp = stack_traversal_output(traversal_around)

            tensorboard.add_image(
                "Latent Traversal Around Z",
                torchvision.utils.make_grid(
                    torch.concat(images_by_row_and_interp), nrow=self.z_dim
                ),
                global_step=self.global_step,
            )

        return total_loss

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

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_lbfgs=False,
    ):
        super().optimizer_step(
            epoch,
            batch_idx,
            optimizer,
            optimizer_idx,
            optimizer_closure,
            on_tpu=on_tpu,
            using_lbfgs=using_lbfgs,
        )

    def on_train_epoch_end(self) -> None:
        k = super().on_train_epoch_end()
        gc.collect()
        return k

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

    # def _foveate_image(self, image: torch.Tensor):
    #     # image: (b, c, h, w)
    #     # filters: (out_h, out_w, rf_h, rf_w)
    #     # out: (b, c, out_h, out_w, rf_h, rf_w)
    #     # return torch.einsum("bchw,ijhw->bcij", image, self.foveation_filters)

    #     b, c, h, w = image.shape

    #     if self.foveation_padding_mode == "replicate":
    #         padding_mode = "replicate"
    #         pad_value = None
    #     elif self.foveation_padding_mode == "zeros":
    #         padding_mode = "constant"
    #         pad_value = 0.0
    #     else:
    #         raise ValueError(f"Unknown padding mode: {self.foveation_padding_mode}")

    #     if self.foveation_padding == "max":
    #         padded_image = F.pad(
    #             image,
    #             (h, h, w, w),
    #             mode=padding_mode,
    #             value=pad_value,
    #         )
    #     elif self.foveation_padding > 0:
    #         padded_image = F.pad(
    #             image,
    #             (
    #                 self.foveation_padding,
    #                 self.foveation_padding,
    #                 self.foveation_padding,
    #                 self.foveation_padding,
    #             ),
    #             mode=padding_mode,
    #             value=pad_value,
    #         )
    #     else:
    #         padded_image = image

    #     pad_h, pad_w = padded_image.shape[-2:]

    #     filter_rf_x, filter_rf_y = self.foveation_filters.shape[-2:]

    #     return F.conv3d(
    #         padded_image.view(b, 1, c, pad_h, pad_w),
    #         self.foveation_filters.view(-1, 1, 1, filter_rf_x, filter_rf_y),
    #         padding=0,
    #         stride=1,
    #     )
