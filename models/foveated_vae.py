from collections import defaultdict
import gc
from copy import deepcopy
import os
import time
from timeit import default_timer as timer
from typing import Any, Iterable, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
from torch import nn, optim

torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2

import wandb
from wandb.sdk.data_types.image import MEDIA_TMP

import utils.foveation as fov_utils
from data import ImageDataModule
from modules.lvae import Ladder, LadderVAE, NextPatchPredictor
from utils.misc import recursive_detach, recursive_to
from utils.vae_utils import (
    free_bits_kl,
    gaussian_kl_divergence,
    gaussian_likelihood,
    reparam_sample,
)
from utils.visualization import (
    fig_to_nparray,
    imshow_unnorm,
    plot_gaussian_foveation_parameters,
    plot_layer_kl_history_by_dim,
    visualize_model_output,
    plot_heatmap,
)

# from memory_profiler import profile
plt.ioff()


class FoVAE(pl.LightningModule):
    def __init__(
        self,
        image_dim=28,
        fovea_radius=2,
        patch_dim=6,
        patch_channels=3,
        patch_ring_scaling_factor=2.0,
        patch_max_ring_radius=None,  # None defaults to half of image
        num_steps: int = 1,
        ladder_dims: List[int] = [25],
        z_dims: List[int] = [10],
        ladder_hidden_dims: List[List[int]] = [[256, 256]],
        lvae_inf_hidden_dims: List[List[int]] = [[256, 256]],
        lvae_gen_hidden_dims: List[List[int]] = [[256, 256]],
        npp_embed_dim: int = 256,
        npp_hidden_dim: int = 512,
        npp_num_heads: int = 1,
        npp_num_layers: int = 3,
        foveation_padding: Union[Literal["max"], int] = "max",
        foveation_padding_mode: Literal["zeros", "replicate"] = "replicate",
        lr=1e-3,
        betas: dict = dict(
            curr_patch_recon=1,
            curr_patch_kl=1,
            next_patch_pos_kl=1,
            next_patch_recon=1,
            next_patch_kl=1,
            image_recon=1,
            spectral_norm=0,
        ),
        free_bits_kl=0,
        soft_foveation_grid_size=None,
        soft_foveation_sigma=0.1,
        soft_foveation_local_bias=1000.0,
        do_soft_foveation=False,
        # n_spectral_iter=1,
        grad_skip_threshold=-1,
        do_batch_norm=False,
        do_weight_norm=False,
        do_gen_skip_connection=False,
        # do_use_beta_norm=True,
        frac_random_foveation=0.0,
        do_image_reconstruction=True,
        do_next_patch_prediction=True,
        reconstruct_fovea_only=False,
        do_lateral_connections=True,
        do_sigmoid_next_location=False,
        npp_do_mask_to_last_step=False,
        npp_do_flag_last_step=False,
        npp_do_curiosity=False,
        image_reconstruction_frac=1.0,
    ):
        super().__init__()

        self.image_dim = image_dim
        self.fovea_radius = fovea_radius
        self.patch_dim = patch_dim

        assert (
            len(ladder_dims)
            == len(z_dims)
            == len(ladder_hidden_dims)
            == len(lvae_inf_hidden_dims)
            == len(lvae_gen_hidden_dims)
        ), "Layer specifications must all have the same length"

        self.num_vae_levels = len(ladder_dims)

        self.z_dims = z_dims

        self.num_steps = num_steps
        self.num_channels = patch_channels + 2
        self.lr = lr
        self.foveation_padding = foveation_padding
        self.foveation_padding_mode = foveation_padding_mode

        # left/right singular vectors used for SR
        self.n_spectral_power_iter = 1  # n_spectral_iter
        self.sr_u = {}
        self.sr_v = {}

        input_dim = self.num_channels * self.patch_dim * self.patch_dim

        self.ladder = Ladder(
            input_dim,
            ladder_dims,
            ladder_hidden_dims,
            batch_norm=do_batch_norm,
            weight_norm=do_weight_norm,
            skip_connection=False,  # TODO
        )
        self.ladder_vae = LadderVAE(
            input_dim,
            ladder_dims,
            z_dims,
            lvae_inf_hidden_dims,
            lvae_gen_hidden_dims,
            batch_norm=do_batch_norm,
            weight_norm=do_weight_norm,
            gen_skip_connection=do_gen_skip_connection,
        )
        self.next_patch_predictor = NextPatchPredictor(
            image_dim=image_dim,
            ladder_vae=self.ladder_vae,
            z_dims=z_dims,
            embed_dim=npp_embed_dim,
            hidden_dim=npp_hidden_dim,
            num_heads=npp_num_heads,
            num_layers=npp_num_layers,
            do_lateral_connections=do_lateral_connections,
            do_sigmoid_next_location=do_sigmoid_next_location,
            do_flag_last_step=npp_do_flag_last_step,
        )
        # self.patch_noise_std = nn.Parameter(torch.ones(input_dim), requires_grad=True)
        # self.patch_noise_std = nn.Parameter(torch.tensor([np.sqrt(1/12)]), requires_grad=True)

        # value based on Theis et al. 2016 “A Note on the Evaluation of Generative Models.”
        # uniform noise with std of 1/12, scaled from being appropriate for input [0,255] to [-1,1]
        self.patch_noise_std = nn.Parameter(
            torch.tensor([np.sqrt(1 / 12 / 127.5)]), requires_grad=False
        )

        self.ff_layers = []
        # self.all_conv_layers = []
        # self.all_bn_layers = []
        for n, layer in self.named_modules():
            # if isinstance(layer, Conv2D) and '_ops' in n:   # only chose those in cell
            if isinstance(layer, nn.Linear):
                self.ff_layers.append(layer)
            # if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.SyncBatchNorm) or \
            #         isinstance(layer, SyncBatchNormSwish):
            #     self.all_bn_layers.append(layer)

        # self._beta = beta

        self.grad_skip_threshold = grad_skip_threshold

        # # TODO
        # if do_use_beta_norm:
        #     beta_vae = (beta * z_dims[0]) / input_dim  # according to beta-vae paper
        #     print(
        #         f"Using normalized betas[1] value of {beta_vae:.6f} as beta, "
        #         f"calculated from unnormalized beta_vae {beta:.6f}"
        #     )
        # else:
        #     beta_vae = beta

        self.betas = betas

        self.free_bits_kl = free_bits_kl

        # image: (b, c, image_dim[0], image_dim[1])
        # TODO: sparsify
        self.default_gaussian_filter_params = fov_utils.get_default_gaussian_foveation_filter_params(
            image_dim=(image_dim, image_dim),
            fovea_radius=fovea_radius,
            image_out_dim=patch_dim,
            # in pyramidal case, pixel ring i averages ring_scaling_factor^i pixels
            ring_sigma_scaling_factor=patch_ring_scaling_factor,
            max_ring_radius=patch_max_ring_radius,
            device=self.device,
        )
        self.frac_random_foveation = frac_random_foveation
        self.do_image_reconstruction = do_image_reconstruction
        self.do_next_patch_prediction = do_next_patch_prediction
        self.reconstruct_fovea_only = reconstruct_fovea_only
        self.image_reconstruction_fraction = image_reconstruction_frac
        self.do_lateral_connections = do_lateral_connections
        self.npp_do_mask_to_last_step = npp_do_mask_to_last_step
        self.npp_do_curiosity = npp_do_curiosity
        self.do_soft_foveation = do_soft_foveation
        if do_soft_foveation:
            self.soft_foveation_grid_size = soft_foveation_grid_size
            self.soft_foveation_sigma = nn.Parameter(
                torch.tensor([soft_foveation_sigma]).float(), requires_grad=False
            )
            self.soft_foveation_local_bias = nn.Parameter(
                torch.tensor([soft_foveation_local_bias]).float(), requires_grad=False
            )

        # Disable automatic optimization!
        # self.automatic_optimization = False

        self.save_hyperparameters()

    def to(self, *args, **kwargs):
        if ("device" in kwargs and kwargs["device"].type == "mps") or any(
            [(isinstance(a, torch.device) and a.type == "mps") for a in args]
        ):
            # not sure why this is necessary
            kwargs["dtype"] = torch.float32
        g = super().to(*args, **kwargs)
        g.default_gaussian_filter_params = recursive_to(
            self.default_gaussian_filter_params, *args, **kwargs
        )
        return g

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        b, c, h, w = x.size()

        x_full = self._add_pos_encodings_to_img_batch(x)

        DO_KL_ON_INPUT_LEVEL = False
        DO_KL_AGAINST_NEXT_LEVEL = True

        curr_patch_rec_total_loss = torch.tensor(0.0, device=x.device)
        curr_patch_kl_div_total_loss = torch.tensor(0.0, device=x.device)
        next_patch_pos_kl_div_total_loss = torch.tensor(0.0, device=x.device)
        next_patch_rec_total_loss = torch.tensor(0.0, device=x.device)
        next_patch_kl_div_total_loss = torch.tensor(0.0, device=x.device)
        image_reconstruction_loss = torch.tensor(0.0, device=x.device)
        curr_patch_kl_divs_by_layer, next_patch_rec_losses_by_layer, next_patch_kl_divs_by_layer = (
            [None] * (self.num_vae_levels + 1),
            [None] * (self.num_vae_levels + 1),
            [None] * (self.num_vae_levels + 1),
        )

        def memoized_patch_getter(x_full, return_full_periphery=False):
            _fov_memo = None

            def get_patch_from_pos(next_pos_x_dist, next_pos_y_dist):
                # TODO: investigate why reshape vs. view is needed
                nonlocal _fov_memo
                patch, _fov_memo = self._foveate_to_loc(
                    x_full,
                    next_pos_x_dist,
                    next_pos_y_dist,
                    do_soft_foveation=self.do_soft_foveation,
                    _fov_memo=_fov_memo,
                )
                patch = patch.reshape(b, -1)
                assert patch.shape == (b, self.num_channels * self.patch_dim * self.patch_dim)
                if return_full_periphery:
                    return patch, _fov_memo["pyramid"][-1]
                else:
                    return patch

            return get_patch_from_pos

        get_patch_from_pos = memoized_patch_getter(x_full)

        initial_x_pos_dist = torch.zeros((b, self.image_dim), device=x.device)
        initial_x_pos_dist[:, self.image_dim // 2] = 1.0
        initial_y_pos_dist = torch.zeros((b, self.image_dim), device=x.device)
        initial_y_pos_dist[:, self.image_dim // 2] = 1.0
        # initial_positions = torch.tensor([0.0, 0.0], device=x.device).unsqueeze(0).repeat(b, 1)
        patches = [get_patch_from_pos(initial_x_pos_dist, initial_y_pos_dist)]
        patch_positions = [(initial_x_pos_dist, initial_y_pos_dist)]

        real_patch_zs = []
        real_patch_dicts = []

        gen_patch_zs = []
        gen_patch_dicts = []

        for step in range(self.num_steps):
            curr_patch = patches[-1]
            curr_patch_dict = self._process_patch(curr_patch)
            # curr_patch_dict:
            #   mu_stds_inference: list(n_vae_layers) of (mu, std) tuples, each (b, z_dim)
            #   mu_stds_gen_prior: list(n_vae_layers+1) of (mu, std) tuples, each (b, z_dim)
            #   mu_stds_gen: list(n_vae_layers+1) of (mu, std) tuples, each (b, z_dim)
            #   sample_zs: list(n_vae_layers+1) of (b, z_dim)
            # each list is in order from bottom to top.
            # gen lists and sample_zs have input-level at index 0
            assert torch.is_same_size(curr_patch, curr_patch_dict["sample_zs"][0])
            real_patch_zs.append(curr_patch_dict["sample_zs"])
            real_patch_dicts.append(curr_patch_dict)

            do_random_foveation = torch.rand(b, device=x.device) < self.frac_random_foveation

            if self.do_next_patch_prediction:
                next_patch_dict = self._gen_next_patch(
                    real_patch_zs,
                    curr_patch_ladder_outputs=curr_patch_dict["ladder_outputs"]
                    if self.do_lateral_connections
                    else None,
                    randomize_next_location=do_random_foveation,
                    mask_to_last_step=self.npp_do_mask_to_last_step,
                )
                # next_patch_dict:
                #   generation:
                #     mu_stds_gen_prior: list(n_vae_layers+1) of (mu, std) tuples,
                #                                                   each (b, z_dim)
                #     mu_stds_gen: list(n_vae_layers+1) of (mu, std) tuples, each (b, z_dim)
                #     sample_zs: list(n_vae_layers+1) of (b, z_dim)
                #   position:
                #     next_pos: (b, 2)
                #     next_pos_mu: (b, 2)
                #     next_pos_std: (b, 2)
                gen_patch_zs.append(next_patch_dict["generation"]["sample_zs"])
                gen_patch_dicts.append(next_patch_dict)

                next_pos_x_dist = next_patch_dict["position"]["next_pos_x_dist"]
                next_pos_y_dist = next_patch_dict["position"]["next_pos_y_dist"]
                # next_pos_offset = next_patch_dict["position"]["next_pos"]
            elif self.frac_random_foveation == 1.0:
                next_pos_x_dist, next_pos_y_dist = self._get_random_foveation_pos(batch_size=b)
                # next_pos_offset = self._get_random_foveation_pos(batch_size=b)
            else:
                raise ValueError(
                    "Must do either next patch prediction or frac_random_foveation=1.0"
                )

            # if True: # do relative positions
            #     next_pos = patch_positions[-1] + next_pos_offset
            #     # sigmoid to keep positions in range [-1, 1]
            #     next_pos = torch.sigmoid(next_pos) * 2 - 1

            # # foveate to next position
            # if torch.isnan(next_pos).any():
            #     next_pos = torch.nan_to_num(next_pos, nan=0.0, posinf=0.0, neginf=0.0)

            next_patch = get_patch_from_pos(next_pos_x_dist, next_pos_y_dist)
            assert torch.is_same_size(next_patch, curr_patch)
            patches.append(next_patch)
            patch_positions.append((next_pos_x_dist, next_pos_y_dist))

            # calculate losses

            # calculate rec and kl losses for current patch
            _curr_patch_rec_loss = -1 * self._patch_likelihood(
                curr_patch,
                mu=curr_patch_dict["sample_zs"][0],
                std=self.patch_noise_std,
                is_bottom_level=True,
                fovea_only=self.reconstruct_fovea_only,
            )
            curr_patch_rec_total_loss += _curr_patch_rec_loss

            _curr_patch_mu_std = curr_patch_dict["mu_stds_gen"]
            for i, level_kl in enumerate(
                _curr_patch_kls_by_layer := self._compute_layerwise_kl(
                    _curr_patch_mu_std,
                    do_kl_on_input_level=DO_KL_ON_INPUT_LEVEL,
                    do_kl_against_next_level=DO_KL_AGAINST_NEXT_LEVEL,
                )
            ):
                if curr_patch_kl_divs_by_layer[i] is None:
                    curr_patch_kl_divs_by_layer[i] = level_kl
                else:
                    curr_patch_kl_divs_by_layer[i] += level_kl

            # calculate kl divergence between predicted next patch pos and std-normal prior
            # only do kl divergence because
            # reconstruction of next_pos is captured in next_patch_rec_loss
            _next_patch_pos_kl_div = 0.0
            # if self.frac_random_foveation < 1.0:
            #     _next_patch_pos_kl_div = self._kl_divergence(
            #         mu=next_patch_dict["position"]["next_pos_mu"],
            #         std=next_patch_dict["position"]["next_pos_std"],
            #     ).sum(-1)
            next_patch_pos_kl_div_total_loss += _next_patch_pos_kl_div

            # if any previous predicted patch, calculate loss between
            # current patch and previous predicted patch
            if self.do_next_patch_prediction and len(gen_patch_zs) > 1:
                # -2 because -1 is the current step, and -2 is the previous step
                prev_gen_patch_dict = gen_patch_dicts[-2]
                prev_gen_mu_std = prev_gen_patch_dict["generation"]["mu_stds_gen"]
                for i, (mu, std) in enumerate(prev_gen_mu_std):
                    if i == 0:
                        # input-level, compare against real patch
                        level_rec_loss = -1 * self._patch_likelihood(
                            curr_patch,
                            mu=mu,  # TODO: curr_patch_dict["sample_zs"][0]???
                            std=self.patch_noise_std,
                            is_bottom_level=True,
                            fovea_only=self.reconstruct_fovea_only,
                        )
                    else:
                        level_rec_loss = -1 * self._patch_likelihood(
                            curr_patch_dict["sample_zs"][i],
                            mu=mu,
                            std=std,
                        )

                    if next_patch_rec_losses_by_layer[i] is None:
                        next_patch_rec_losses_by_layer[i] = level_rec_loss
                    else:
                        next_patch_rec_losses_by_layer[i] += level_rec_loss

                for i, level_kl in enumerate(
                    _next_patch_kls_by_layer := self._compute_layerwise_kl(
                        prev_gen_mu_std,
                        do_kl_on_input_level=DO_KL_ON_INPUT_LEVEL,
                        do_kl_against_next_level=DO_KL_AGAINST_NEXT_LEVEL,
                    )
                ):
                    if next_patch_kl_divs_by_layer[i] is None:
                        next_patch_kl_divs_by_layer[i] = level_kl
                    else:
                        next_patch_kl_divs_by_layer[i] += level_kl

            # check that the next patch's position is close to the position from which
            # it was supposed to be extracted
            # position will wobble a little due to gaussian aggregation? TODO: investigate
            # I don't care as long as it's under half a pixel

            # this is commented because we do not allow foveation outside the image, and
            # depending on method of padding, locations in the padding area will not
            # have true locations, but will be clamped to locations at the edge of
            # the image (or zeros). As such, averaging the locations on the patch
            # has to be inside the image, but the location would encompass locations
            # outside the image. this is a useful check though, and should be re-enabled
            # if the foveation method is changed to not rely on padding and/or clamping
            # _next_patch_center = next_patch.view(
            #     b, self.num_channels, self.patch_dim, self.patch_dim
            # )[:, -2:, :, :].mean(dim=(2, 3))
            # _acceptable_dist_threshold = 0.5 / min(h, w)
            # assert (
            #     _next_patch_center - next_patch_sample_pos
            # ).abs().max() <= _acceptable_dist_threshold, (
            #     f"Next patch position {_next_patch_center.round(2).cpu()} is too "
            #     f"far from predicted position {next_patch_sample_pos.round(2).cpu()}: "
            #     f"{(_next_patch_center - next_patch_sample_pos).abs().max()} > "
            #     f"{_acceptable_dist_threshold}"
            # )

        if self.do_image_reconstruction:
            image_reconstruction_loss, _ = self._reconstruct_image(
                real_patch_zs,
                x_full,
                return_patches=False,
                fovea_only=True,  # self.reconstruct_fovea_only,
                proportion=self.image_reconstruction_fraction,
            )
        else:
            image_reconstruction_loss = torch.tensor(0.0, device=self.device)

        # TODO: there's a memory leak somewhere, comes out during overfit_batches=1
        # Notes on memory leak:
        # https://github.com/Lightning-AI/lightning/issues/16876
        # https://github.com/pytorch/pytorch/issues/13246
        # https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/

        DO_COMPUTE_NEXT_PATCH_LOSSES = self.do_next_patch_prediction and self.num_steps > 1
        # aggregate losses across steps
        # mean over steps
        curr_patch_kl_divs_by_layer = [g / self.num_steps for g in curr_patch_kl_divs_by_layer]
        if DO_COMPUTE_NEXT_PATCH_LOSSES:
            next_patch_rec_losses_by_layer = [
                g / self.num_steps for g in next_patch_rec_losses_by_layer
            ]
            next_patch_kl_divs_by_layer = [g / self.num_steps for g in next_patch_kl_divs_by_layer]

        curr_patch_rec_total_loss = (
            self.betas["curr_patch_recon"] * curr_patch_rec_total_loss / self.num_steps
        )
        # sum over layers, mean over z_dim (already mean over steps)
        # TODO: apply free bits to sum of z_dim KLs as opposed to every dim as now?
        curr_patch_kl_div_total_loss = (
            self.betas["curr_patch_kl"]
            * torch.stack(
                [
                    free_bits_kl(g, self.free_bits_kl).mean(dim=0)
                    for g in curr_patch_kl_divs_by_layer
                ],
                dim=0,
            ).sum()
        )
        next_patch_pos_kl_div_total_loss = self.betas["next_patch_pos_kl"] * free_bits_kl(
            next_patch_pos_kl_div_total_loss / self.num_steps, self.free_bits_kl
        )
        # mean over z_dim, sum over layers (already mean over steps)
        if DO_COMPUTE_NEXT_PATCH_LOSSES:
            next_patch_rec_total_loss = (
                self.betas["next_patch_recon"]
                * torch.stack(next_patch_rec_losses_by_layer, dim=0).sum()
            )
            next_patch_kl_div_total_loss = (
                self.betas["next_patch_kl"]
                * torch.stack(
                    [
                        free_bits_kl(g, self.free_bits_kl).mean(dim=0)
                        for g in next_patch_kl_divs_by_layer
                    ],
                    dim=0,
                ).sum()
            )

        image_reconstruction_loss = self.betas["image_recon"] * image_reconstruction_loss
        spectral_norm = (
            self.betas["spectral_norm"] * self.spectral_norm_parallel()
            if self.betas["spectral_norm"] > 0
            else torch.tensor(0.0, device=self.device)
        )

        total_loss = (
            curr_patch_rec_total_loss
            + curr_patch_kl_div_total_loss
            + next_patch_pos_kl_div_total_loss
            + next_patch_rec_total_loss
            + next_patch_kl_div_total_loss
            + image_reconstruction_loss
            + spectral_norm
        )

        # detach auxiliary outputs
        curr_patch_kl_divs_by_layer = recursive_detach(curr_patch_kl_divs_by_layer)
        if DO_COMPUTE_NEXT_PATCH_LOSSES:
            next_patch_rec_losses_by_layer = recursive_detach(next_patch_rec_losses_by_layer)
            next_patch_kl_divs_by_layer = recursive_detach(next_patch_kl_divs_by_layer)
        patches = recursive_detach(patches)
        patch_positions = recursive_detach(patch_positions)
        real_patch_zs = recursive_detach(real_patch_zs)
        real_patch_dicts = recursive_detach(real_patch_dicts)
        gen_patch_zs = recursive_detach(gen_patch_zs)
        gen_patch_dicts = recursive_detach(gen_patch_dicts)

        return dict(
            losses=dict(
                total_loss=total_loss,
                curr_patch_total_loss=curr_patch_rec_total_loss + curr_patch_kl_div_total_loss,
                curr_patch_rec_loss=curr_patch_rec_total_loss,
                curr_patch_kl_loss=curr_patch_kl_div_total_loss,
                next_patch_pos_kl_loss=next_patch_pos_kl_div_total_loss,
                next_patch_rec_loss=next_patch_rec_total_loss,
                next_patch_kl_loss=next_patch_kl_div_total_loss,
                image_reconstruction_loss=image_reconstruction_loss,
                spectral_norm=spectral_norm,
            ),
            losses_by_layer=dict(
                curr_patch_kl_divs_by_layer=curr_patch_kl_divs_by_layer,  # n_levels, z_dim
                next_patch_rec_losses_by_layer=next_patch_rec_losses_by_layer,  # n_levels
                next_patch_kl_divs_by_layer=next_patch_kl_divs_by_layer,  # n_levels, z_dim
            ),
            step_vars=dict(
                patches=patches,  # n_steps x (b, n_channels, patch_dim, patch_dim)
                patch_positions=patch_positions,  # n_steps x (b, 2)
                real_patch_zs=real_patch_zs,  # n_steps x n_levels x (b, z_dim)
                real_patch_dicts=real_patch_dicts,  # n_steps x n_levels x dict
                gen_patch_zs=gen_patch_zs,  # n_steps x n_levels x (b, z_dim)
                gen_patch_dicts=gen_patch_dicts  # n_steps x n_levels x dict
                # (b, n_patches, n_channels, patch_dim, patch_dim)
                # image_reconstruction_patches=image_reconstruction_patches,
            ),
        )

    def _compute_layerwise_kl(
        self,
        mu_stds_gen: Tuple[torch.Tensor, torch.Tensor],
        do_kl_on_input_level=False,
        do_kl_against_next_level=True,
    ):
        kl_divs_by_layer = []
        for i, (mu, std) in enumerate(mu_stds_gen):
            if 0 < i < len(mu_stds_gen) - 1 and do_kl_against_next_level:
                # not top-level, compare against next-level mu_prior
                mu_prior, std_prior = mu_stds_gen[i + 1]
            else:
                # top-level or bottom-level, compare against std-normal prior
                mu_prior, std_prior = torch.zeros_like(mu), torch.ones_like(std)

            level_kl = self._kl_divergence(mu=mu, std=std, mu_prior=mu_prior, std_prior=std_prior)

            if i == 0 and not do_kl_on_input_level:
                level_kl = torch.zeros_like(level_kl)

            kl_divs_by_layer.append(level_kl)
        return kl_divs_by_layer

    def _patch_likelihood(self, patch, mu, std, is_bottom_level=False, fovea_only=False):
        if std.size() == torch.Size([1]):
            std = std.expand(mu.size())

        if is_bottom_level and fovea_only:
            patch = self._patch_to_fovea(patch)
            mu = self._patch_to_fovea(mu)
            std = self._patch_to_fovea(std)

        return gaussian_likelihood(
            patch,
            mu=mu,
            std=std,
            batch_reduce_fn="mean",
            # norm_std_method="bounded" if is_bottom_level else "explin",
            # norm_std_bound_min=self.patch_noise_std if is_bottom_level else None,
            # norm_std_bound_max=1.0 if is_bottom_level else None,
        )

    def _kl_divergence(self, mu, std, mu_prior=0.0, std_prior=1.0, batch_reduce_fn="mean"):
        if std.size() == torch.Size([1]):
            std = std.expand(mu.size())

        return gaussian_kl_divergence(
            mu=mu,
            std=std,
            mu_prior=mu_prior,
            std_prior=std_prior,
            batch_reduce_fn=batch_reduce_fn,
        )

    def _get_random_foveation_pos(self, batch_size: int):
        x = self.next_patch_predictor._get_random_foveation_pos(batch_size, device=self.device)
        y = self.next_patch_predictor._get_random_foveation_pos(batch_size, device=self.device)
        return x, y

    def _reconstruct_image(
        self,
        sample_zs,
        image: Optional[torch.Tensor],
        return_patches=False,
        fovea_only=False,
        proportion=1.0,
    ):
        # positions span [-1, 1] in both x and y

        b = sample_zs[0][0].size(0)

        positions_x = torch.linspace(
            -1, 1, steps=int(np.ceil(self.image_dim / (self.fovea_radius * 2))), device="cpu"
        )
        positions_y = torch.linspace(
            -1, 1, steps=int(np.ceil(self.image_dim / (self.fovea_radius * 2))), device="cpu"
        )
        positions = torch.stack(
            torch.meshgrid(positions_x, positions_y, indexing="xy"), dim=-1
        ).view(-1, 2)

        if proportion < 1.0:
            # sample positions
            n_positions = positions.size(0)
            n_to_sample = int(np.ceil(n_positions * proportion))
            sampled_indices = [torch.randperm(n_positions)[:n_to_sample] for _ in range(b)]
            sampled_indices = torch.stack(sampled_indices, dim=1).view(-1)
            sampled_positions = positions[sampled_indices].view(-1, b, 2).contiguous()
        else:
            sampled_positions = positions.unsqueeze(1).expand(-1, b, -1).contiguous()

        def memoized_patch_getter(image):
            _fov_memo = None

            def get_patch_from_pos(pos):
                nonlocal _fov_memo
                # disable soft foveation for getting real patches to reconstruct,
                # regardless of whether the model is using soft-foveation
                next_pos_x_dist, next_pos_y_dist = self._get_dummy_dist_for_pos(pos)
                patch, _fov_memo = self._foveate_to_loc(
                    image, next_pos_x_dist, next_pos_y_dist, do_soft_foveation=False, _fov_memo=_fov_memo
                )
                return patch

            return get_patch_from_pos

        # predict zs for each position
        image_recon_loss = None
        gen_zs = [*sample_zs]
        patches = []
        if image is not None:
            _memo_foveate_to_loc = memoized_patch_getter(image)
        # TODO: maybe reconstruct only some patches?

        for i, position in enumerate(sampled_positions):
            position = position.to(self.device)
            # TODO: consider passing mu stds from prev patch?
            gen_dict = self._gen_next_patch(
                gen_zs, forced_next_location=position, mask_to_last_step=False
            )
            gen_patch = gen_dict["generation"]["sample_zs"][0]
            # gen_mu, gen_std = gen_dict["generation"]["mu_stds_gen"][0]
            # gen_zs.append(gen_dict["generation"]["sample_zs"])
            gen_patch = gen_patch.view(b, self.num_channels, self.patch_dim, self.patch_dim)
            if image is not None:
                if image_recon_loss is None:
                    image_recon_loss = 0.0
                real_patch = _memo_foveate_to_loc(position)
                assert torch.is_same_size(gen_patch, real_patch)

                patch_recon_loss = -1 * self._patch_likelihood(
                    real_patch.view(b, -1),
                    mu=gen_patch.view(b, -1),
                    std=self.patch_noise_std,
                    is_bottom_level=True,
                    fovea_only=fovea_only,
                )
                # patch_recon_loss = -1 * self._patch_likelihood(
                #     real_patch.view(b, -1), mu=gen_mu, std=torch.ones_like(gen_mu), fovea_only=fovea_only
                # )
                image_recon_loss += patch_recon_loss

            patches.append(self._patch_to_fovea(gen_patch) if fovea_only else gen_patch)

        if image_recon_loss is not None:
            image_recon_loss /= sampled_positions.size(0)

        if return_patches:
            return image_recon_loss, torch.stack(patches, dim=0).transpose(0, 1)
        else:
            return image_recon_loss, None

    def _foveate_to_loc(
        self,
        image: torch.Tensor,
        next_pos_x_dist: torch.Tensor,
        next_pos_y_dist: torch.Tensor,
        do_soft_foveation=None,
        _fov_memo: dict = None,
    ):
        # image: (b, c, h, w)
        # loc: (b, 2), where entries are in [-1, 1]
        # filters: (out_h, out_w, rf_h, rf_w)
        # out: (b, c, out_h, out_w, rf_h, rf_w)

        # start = timer()
        # is_memo = _fov_memo is not None

        b, c, h, w = image.shape

        if self.foveation_padding_mode == "replicate":
            padding_mode = "replicate"
            pad_value = None
        elif self.foveation_padding_mode == "zeros":
            padding_mode = "constant"
            pad_value = 0.0
        else:
            raise ValueError(f"Unknown padding mode: {self.foveation_padding_mode}")

        if _fov_memo is not None and _fov_memo["orig_image"] is image:
            padded_image = _fov_memo["padded_image"]
            pad_offset = _fov_memo["pad_offset"]
        else:
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

        if True:  # do_soft_foveation:
            if _fov_memo and "soft_patches" in _fov_memo:
                soft_patches = _fov_memo["soft_patches"]
                all_locs = _fov_memo["all_locs"]
            else:
                # build patches for all possible locations
                # build grid of locations spanning (-1, 1) in both dims
                # (out_h, out_w, 2)
                # all_locs = torch.stack(
                #     torch.meshgrid(
                #         torch.linspace(-1, 1, self.soft_foveation_grid_size),
                #         torch.linspace(-1, 1, self.soft_foveation_grid_size),
                #     ),
                #     dim=-1,
                # ).to(self.device)
                all_locs = torch.stack(
                    torch.meshgrid(
                        torch.linspace(-1, 1, self.image_dim),
                        torch.linspace(-1, 1, self.image_dim),
                    ),
                    dim=-1,
                ).to(self.device)
                # (out_h, out_w, 2) -> (b*out_h*out_w, 2)
                all_locs = all_locs.view(-1, 1, 2).repeat(1, b, 1)
                soft_patches = []
                for _loc in all_locs:
                    # consider vectorizing
                    gaussian_filter_params = self._move_default_filter_params_to_loc(
                        _loc, (h, w), pad_offset
                    )
                    _patches, _fov_memo = fov_utils.apply_mean_foveation_pyramid(
                        padded_image, gaussian_filter_params, memo=_fov_memo
                    )
                    soft_patches.append(_patches)
                # (out_h * out_w, b, c, h, w) -> (b, out_h * out_w, c, h, w)
                soft_patches = torch.stack(soft_patches, dim=0).transpose(0, 1)

            # # soft attention over soft_patches based on position encodings (loc vs. last 2 dimensions of c)
            # # loc: (b, 2)
            # # soft_patches: (b, out_h*out_w, c, h, w)
            # # foveated_image: (b, c, h, w)
            # _loc = (
            #     torch.cat(
            #         (
            #             torch.zeros((b, self.num_channels - 2), device=self.device),
            #             loc,
            #         ),
            #         dim=-1,
            #     )
            #     .view(b, c, 1, 1)
            #     .expand(b, c, self.patch_dim, self.patch_dim)
            # )

            # # mask out all channels except the last 2 (position)
            # if _fov_memo and "channel_mask" in _fov_memo:
            #     channel_mask = _fov_memo["channel_mask"]
            # else:
            #     channel_mask = torch.zeros((1, self.num_channels, 1, 1), device=self.device)
            #     channel_mask[:, -2:, :, :] = 1.0

            # # make gaussian attention mask over all_locs, centered at loc
            # # TODO: soft_foveation_sigma
            # gaussian_mask = (
            #     F.softmax(
            #         torch.exp(
            #             -torch.sum((all_locs - loc.unsqueeze(0)) ** 2, dim=-1)
            #             / (2 * self.soft_foveation_sigma**2)
            #         ),
            #         dim=0,
            #     )
            #     .transpose(0, 1)
            #     .unsqueeze(1)
            # )

            # soft attention over soft_patches
            # soft_patches: (b, image_dim*image_dim, c, h, w)
            # foveated_image: (b, c, h, w)
            # weights given by cross product of x and y distributions
            weights = (next_pos_x_dist.unsqueeze(1) * next_pos_y_dist.unsqueeze(2)).view(b, -1)
            assert weights.size() == torch.Size([b, self.image_dim * self.image_dim])
            foveated_image = torch.einsum("bg,bgchw->bchw", weights, soft_patches)

            # embed_dim = c * self.patch_dim * self.patch_dim
            # foveated_image = F.scaled_dot_product_attention(
            #     (_loc * channel_mask).view(b, 1, embed_dim),
            #     (soft_patches * channel_mask.unsqueeze(1)).view(b, -1, embed_dim),
            #     soft_patches.view(b, -1, embed_dim),
            #     attn_mask=gaussian_mask * self.soft_foveation_local_bias,
            # )
            foveated_image = foveated_image.view(b, c, self.patch_dim, self.patch_dim)

        else:
            # move the gaussian filter params to the loc
            gaussian_filter_params = self._move_default_filter_params_to_loc(
                loc, (h, w), pad_offset
            )

            # foveate
            foveated_image, _fov_memo = fov_utils.apply_mean_foveation_pyramid(
                padded_image, gaussian_filter_params, memo=_fov_memo
            )

        _fov_memo["orig_image"] = image
        _fov_memo["padded_image"] = padded_image
        _fov_memo["pad_offset"] = pad_offset
        if do_soft_foveation:
            _fov_memo["soft_patches"] = soft_patches
            _fov_memo["all_locs"] = all_locs
            # _fov_memo["channel_mask"] = channel_mask

        if foveated_image.isnan().any():
            raise ValueError("NaNs in foveated image!")

        # end = timer()
        # print(f"Foveation time (memoized: {is_memo}, training: {self.training}): {end - start}", flush=True)

        return foveated_image, _fov_memo

    def _move_default_filter_params_to_loc(
        self, loc: torch.Tensor, image_dim: Iterable, pad_offset: Optional[Iterable] = None
    ):
        """Move gaussian foveation params centers to center at loc in the padded image"""
        assert len(image_dim) == 2, f"image_dim must be (h, w), got {image_dim}"
        image_dim = torch.tensor(image_dim, dtype=loc.dtype, device=loc.device)

        if loc.min() < -1 or loc.max() > 1:
            print(f"loc must be in [-1, 1], got {loc.min()} to {loc.max()}")

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
            if (
                not torch.isclose(new_mus.mean(1), loc, atol=1e-4).all()
                and not torch.isnan(new_mus).any()
            ):
                print(
                    f"New gaussian centers after move not close to loc: "
                    f"{new_mus.mean(1)[torch.argmax((new_mus.mean(1) - loc).sum(1), 0)]} "
                    f"vs {loc[torch.argmax((new_mus.mean(1) - loc).sum(1), 0)]}"
                )
            ring["mus"] = new_mus + pad_offset

        return gaussian_filter_params

    def _process_patch(self, x: torch.Tensor):
        ladder_outputs = self.ladder(x)
        patch_vae_dict = self.ladder_vae(ladder_outputs)
        patch_vae_dict["ladder_outputs"] = ladder_outputs
        return patch_vae_dict

    def _gen_next_patch(
        self,
        prev_zs: List[List[torch.Tensor]],
        curr_patch_ladder_outputs: Optional[List[torch.Tensor]] = None,
        forced_next_location: Optional[torch.Tensor] = None,
        randomize_next_location: Optional[torch.Tensor] = None,
        mask_to_last_step: bool = False,
    ):
        # prev_zs: list(n_steps_so_far) of lists
        #              (n_levels from lowest to highest) of tensors (b, dim)
        # curr_patch_ladder_outputs: list(n_levels from lowest to highest) of tensors (b, d)
        # next_patch_pos: Tensor (b, 2)
        # highest-level z is the last element of the list
        b = prev_zs[0][0].size(0)

        if forced_next_location is not None:
            forced_next_pos_x_dist, forced_next_pos_y_dist = self._get_dummy_dist_for_pos(forced_next_location)
        else:
            forced_next_pos_x_dist, forced_next_pos_y_dist = None, None

        # randomize next location for those that are masked to true in randomize_next_location
        # TODO: move out of here
        # if randomize_next_location is not None:
        #     next_pos_rand = self._get_random_foveation_pos(b, device=device)
        #     next_pos = torch.where(randomize_next_location[:, None], next_pos_rand, next_pos)

        return self.next_patch_predictor(
            prev_zs,
            curr_patch_ladder_outputs=curr_patch_ladder_outputs,
            forced_next_location_x_dist=forced_next_pos_x_dist,
            forced_next_location_y_dist=forced_next_pos_y_dist,
            randomize_next_location=randomize_next_location,
            mask_to_last_step=mask_to_last_step,
        )

    def _get_dummy_dist_for_pos(self, forced_next_location):
        b = forced_next_location.size(0)
        forced_next_pos_x_dist, forced_next_pos_y_dist = torch.zeros(
                (b, self.image_dim), device=forced_next_location.device
            ), torch.zeros((b, self.image_dim), device=forced_next_location.device)
        f = torch.floor(
                forced_next_location * (self.image_dim / 2) + (self.image_dim / 2)
            ).long().clamp(0, self.image_dim - 1)
        forced_next_pos_x_dist[torch.arange(b), f[:, 0]] = 1.0
        forced_next_pos_y_dist[torch.arange(b), f[:, 1]] = 1.0

        return forced_next_pos_x_dist, forced_next_pos_y_dist


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

    def spectral_norm_parallel(self):
        """This method computes spectral normalization for all FF layers in parallel.

        This method should be called after calling the forward method of all the
        FF layers in each iteration.

        Adapted from https://github.com/NVlabs/NVAE
        """
        weights = {}  # a dictionary indexed by the shape of weights
        for l in self.ff_layers:
            weight = l.weight
            weight_mat = weight.view(weight.size(0), -1)
            if weight_mat.shape not in weights:
                weights[weight_mat.shape] = []

            weights[weight_mat.shape].append(weight_mat)

        loss = 0
        device = self.device
        for i in weights:
            weights[i] = torch.stack(weights[i], dim=0)
            with torch.no_grad():
                num_iter = self.n_spectral_power_iter
                if i not in self.sr_u:
                    num_w, row, col = weights[i].shape
                    self.sr_u[i] = F.normalize(
                        torch.ones(num_w, row).normal_(0, 1).to(device), dim=1, eps=1e-3
                    )
                    self.sr_v[i] = F.normalize(
                        torch.ones(num_w, col).normal_(0, 1).to(device), dim=1, eps=1e-3
                    )
                    # increase the number of iterations for the first time
                    num_iter = 10 * self.n_spectral_power_iter

                for j in range(num_iter):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    self.sr_v[i] = F.normalize(
                        torch.matmul(self.sr_u[i].unsqueeze(1), weights[i]).squeeze(1),
                        dim=1,
                        eps=1e-3,
                    )  # bx1xr * bxrxc --> bx1xc --> bxc
                    self.sr_u[i] = F.normalize(
                        torch.matmul(weights[i], self.sr_v[i].unsqueeze(2)).squeeze(2),
                        dim=1,
                        eps=1e-3,
                    )  # bxrxc * bxcx1 --> bxrx1  --> bxr

            sigma = torch.matmul(
                self.sr_u[i].unsqueeze(1), torch.matmul(weights[i], self.sr_v[i].unsqueeze(2))
            )
            loss += torch.sum(sigma)
        return loss

    # # generate n=num images using the model
    # def generate(self, num: int):
    #     self.eval()
    #     z = torch.randn(num, self.z_dim)
    #     with torch.no_grad():
    #         return self._decode_patch(z)[-1].cpu()

    # # returns z for position-augmented patch
    # def get_patch_zs(self, patch_with_pos: torch.Tensor):
    #     assert patch_with_pos.ndim == 4
    #     self.eval()

    #     with torch.no_grad():
    #         zs, mus, stds = self._encode_patch(patch_with_pos)

    #     return zs

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

    def _patch_to_fovea(self, patch: torch.Tensor):
        assert patch.ndim in (2, 4)
        if patch.ndim == 2:
            p = patch.view(-1, self.num_channels, self.patch_dim, self.patch_dim)
        else:
            p = patch

        ring_radius = self.patch_dim // 2 - self.fovea_radius
        fovea_dim = self.fovea_radius * 2

        fovea = p[:, :, ring_radius : (-ring_radius or None), ring_radius : (-ring_radius or None)]
        if patch.ndim == 2:
            return fovea.reshape(-1, self.num_channels * fovea_dim * fovea_dim)
        else:
            return fovea

    def generate_patch_from_z(self, z, z_level=-1):
        if z_level != -1:
            raise NotImplementedError("Only z_level=-1 is supported for now")
        with torch.no_grad():
            return self.ladder_vae.generate(top_z=z)

    def latent_traverse(self, z, z_level=-1, range_limit=3, step=0.5, around_z=False):
        self.eval()
        interpolation = torch.arange(-range_limit, range_limit + 0.1, step)
        samples = []
        with torch.no_grad():
            for row in range(self.z_dims[z_level]):
                row_samples = []
                # copy to CPU to bypass https://github.com/pytorch/pytorch/issues/94390
                interp_z = z.clone().to("cpu")
                for val in interpolation:
                    if around_z:
                        interp_z[:, row] += val
                    else:
                        interp_z[:, row] = val

                    gen_dict = self.generate_patch_from_z(interp_z.to(self.device), z_level=z_level)
                    sample = (
                        gen_dict["sample_zs"][0]
                        .data.cpu()
                        .reshape(z.shape[0], self.num_channels, self.patch_dim, self.patch_dim)
                    )
                    row_samples.append(sample)
                samples.append(row_samples)
        return samples

    def on_fit_start(self):
        self.trainer.logger.watch(self, log_freq=10, log_graph=False)

    def training_step(self, batch, batch_idx):
        x, y = batch
        forward_out = self.forward(x, y)
        # total_loss = forward_out["losses"].pop("total_loss")
        total_loss = forward_out["losses"]["total_loss"]
        # print("Rec loss", forward_out["losses"]["image_reconstruction_loss"])
        # self.log("train_total_loss", total_loss, prog_bar=True)
        self.log_dict({"train/" + k: v.detach().item() for k, v in forward_out["losses"].items()})
        # patch_noise_std_mean = self.patch_noise_std.detach().mean().item()
        # self.log("patch_noise_std_mean", patch_noise_std_mean, logger=True, on_step=True)

        if self.do_soft_foveation:
            self.log("train/soft_foveation_sigma", self.soft_foveation_sigma.detach().item())
            self.log(
                "train/soft_foveation_local_bias", self.soft_foveation_local_bias.detach().item()
            )

        # self._optimizer_step(loss)
        # TODO: skip on grad norm
        skip_update = float(torch.isnan(total_loss))
        if skip_update:
            print("Skipping update!", forward_out["losses"])

        self.log(
            "n_skipped_nan",
            skip_update,
            on_epoch=True,
            on_step=False,
            logger=True,
            # prog_bar=True,
            reduce_fx=torch.sum,
            sync_dist=True,
        )
        # self.log(grad_norm, skip_update, on_epoch=True, logger=True)

        return None if skip_update else total_loss

    # @profile
    def validation_step(self, batch, batch_idx):
        x, y = batch
        forward_out = self.forward(x, y)
        total_loss = forward_out["losses"]["total_loss"]
        self.log_dict({"val/" + k: v.detach().item() for k, v in forward_out["losses"].items()})

        # plot kl divergences by layer on the same plots
        curr_patch_kl_divs_by_layer = forward_out["losses_by_layer"]["curr_patch_kl_divs_by_layer"]
        if not hasattr(self, "_epoch_curr_patch_kl_history"):
            self._epoch_curr_patch_kl_history = [list() for _ in curr_patch_kl_divs_by_layer]
        for i, v in enumerate(curr_patch_kl_divs_by_layer):
            self._epoch_curr_patch_kl_history[i].append(v.detach().cpu().numpy())

        _curr_kl_divs = {
            f"val/curr_patch_kl_l{i}": v.detach().mean().item()
            for i, v in enumerate(curr_patch_kl_divs_by_layer)
        }
        self.log_dict(_curr_kl_divs)

        if self.do_next_patch_prediction:
            next_patch_kl_divs_by_layer = forward_out["losses_by_layer"][
                "next_patch_kl_divs_by_layer"
            ]
            _next_kl_divs = {
                f"val/next_patch_kl_l{i}": v.detach().mean().item()
                for i, v in enumerate(next_patch_kl_divs_by_layer)
            }
            self.log_dict(_next_kl_divs)

            if not hasattr(self, "_epoch_npp_kl_history"):
                self._epoch_npp_kl_history = [list() for _ in next_patch_kl_divs_by_layer]
            for i, v in enumerate(next_patch_kl_divs_by_layer):
                self._epoch_npp_kl_history[i].append(v.detach().cpu().numpy())

        # if batch_idx == 0:
        #     # batch_size = x.size(0)
        #     self.run_generalization_suite(
        #         batch_size=16,
        #         predict_foveation_path=False,
        #         resize_pre=self.trainer.datamodule.resize,
        #     )

        fov_locations_x = torch.stack([g[0].argmax(dim=-1) for g in forward_out["step_vars"]["patch_positions"]], dim=0).cpu()
        fov_locations_y = torch.stack([g[1].argmax(dim=-1) for g in forward_out["step_vars"]["patch_positions"]], dim=0).cpu()

        fov_locations = torch.cat(
            (fov_locations_x.unsqueeze(-1), fov_locations_y.unsqueeze(-1)), dim=-1
        )

        fov_locations = fov_locations.permute(1, 0, 2)  # (b, n_steps, 2)

        if not hasattr(self, "_epoch_fov_locations"):
            self._epoch_fov_locations = fov_locations
            self._epoch_labels = y.cpu()
        else:
            self._epoch_fov_locations = torch.cat((self._epoch_fov_locations, fov_locations), dim=0)
            self._epoch_labels = torch.cat((self._epoch_labels, y.cpu()), dim=0)

        # delete all variables involved
        del (
            # x,
            # y,
            # forward_out,
            curr_patch_kl_divs_by_layer,
            _curr_kl_divs,
            # step_sample_zs,
            # step_next_z_preds,
            # patches,
            # step_patch_positions,
        )

        # print(gc.get_stats())

        # def _recursive_gc_log_tensors(obj):
        #     try:
        #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        #             print(type(obj), obj.size())
        #         elif isinstance(obj, (list, tuple)):
        #             for o in obj:
        #                 _recursive_gc_log_tensors(o)
        #         elif isinstance(obj, dict):
        #             for o in obj.values():
        #                 _recursive_gc_log_tensors(o)
        #     except:
        #         pass

        # for obj in gc.get_objects():
        #     _recursive_gc_log_tensors(obj)
        return total_loss

    def on_validation_epoch_end(self):
        if not hasattr(self, "curr_patch_kl_history"):
            self.curr_patch_kl_history = [list() for _ in self._epoch_curr_patch_kl_history]
            self.val_epochs = []
            return  # don't log anything on validation check
        self.val_epochs.append(self.current_epoch + 1)  # +1 because epoch starts at 0
        for i, v in enumerate(self._epoch_curr_patch_kl_history):
            self.curr_patch_kl_history[i].append(np.stack(v).mean(axis=0))

        _fig = fig_to_nparray(
            plot_layer_kl_history_by_dim(self.curr_patch_kl_history, self.val_epochs)
        )
        self.logger.log_image("curr_patch_kl_history", [_fig])

        if self.do_next_patch_prediction:
            if not hasattr(self, "npp_kl_history"):
                self.npp_kl_history = [list() for _ in self._epoch_npp_kl_history]
            for i, v in enumerate(self._epoch_npp_kl_history):
                self.npp_kl_history[i].append(np.stack(v).mean(axis=0))

            _fig = fig_to_nparray(
                plot_layer_kl_history_by_dim(self.npp_kl_history, self.val_epochs)
            )
            self.logger.log_image("npp_kl_history", [_fig])

        ### plot foveation behavior
        # plot average foveation locations
        fov_locations = self._epoch_fov_locations
        labels = self._epoch_labels
        # plot heatmap of foveation locations
        locations_transformed = (fov_locations + 1) / 2
        locations_transformed *= self.image_dim
        # locations_transformed = (
        #     torch.round(locations_transformed).long().clamp(0, self.image_dim - 1)
        # )

        def _plot_locations(locs, ax=None, title=None):
            # locs: (b, n_steps, 2)
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.get_figure()

            for i in range(locs.size(1)):
                color = plt.cm.rainbow(i / locs.size(1))
                ax.scatter(
                    locs[:, i, 0], locs[:, i, 1], s=1, alpha=min(1, 200 / len(locs)), color=color
                )
            ax.set_xlim(0, self.image_dim - 1)
            ax.set_ylim(0, self.image_dim - 1)
            ax.set_aspect("equal")
            if title:
                ax.set_title(title)
            return fig, ax

        fig, ax = _plot_locations(locations_transformed, title="Foveation Locations (all labels)")
        fig.tight_layout()
        _fig = fig_to_nparray(fig)
        self.logger.log_image("foveation_locations", [_fig])

        # heatmap = torch.zeros((self.image_dim, self.image_dim))
        # for i in range(locations_transformed.size(0)):
        #     heatmap[locations_transformed[i, 1:, 0], locations_transformed[i, 1:, 1]] += 1
        # heatmap /= heatmap.sum()
        # heatmap = heatmap.numpy()
        # _fig = fig_to_nparray(fig)
        # self.logger.log_image("foveation_heatmap", [_fig])

        # group by label
        d = defaultdict(list)
        for label, loc in zip(labels, locations_transformed):
            d[label.item()].append(loc)

        # plot average foveation locations by 10 random labels
        labels_to_plot = np.random.choice(list(d.keys()), 10, replace=False)
        # make 5x2 grid of plots
        fig, axes = plt.subplots(2, 5, figsize=(20, 10))
        for ax, label in zip(axes.flatten(), labels_to_plot):
            avg_x_for_label = torch.stack(
                [x[0] for x in self.trainer.datamodule.dataset_val if x[1] == label]
            ).mean(dim=0)
            locs = torch.stack(d[label])
            _plot_locations(locs, ax=ax, title=f"Label {label}")
            ax.imshow(avg_x_for_label.permute(1, 2, 0).cpu().numpy(), cmap="gray")

        fig.tight_layout()
        _fig = fig_to_nparray(fig)
        self.logger.log_image("foveation_heatmap_by_label", [_fig])

        del (
            self._epoch_curr_patch_kl_history,
            self._epoch_npp_kl_history,
            self._epoch_fov_locations,
            self._epoch_labels,
        )

    def run_generalization_suite(
        self,
        # dataset_name: Literal["mnist", "omniglot"],
        # transform_name: Literal["none", "translate", "scale"],
        batch_size=16,
        resize_pre=False,
        predict_foveation_path: bool = False,
    ):
        """Run a suite of generalization tests on the model.

        datasets: MNIST, Omniglot
        transforms: none, translate, scale
        predict foveation path
        """
        # datasets: MNIST, Omniglot
        # transforms: none, translate, scale
        # predict foveation path

        for dataset_name in ["mnist", "omniglot"]:
            dataset = ImageDataModule(
                dataset=dataset_name, batch_size=batch_size, resize=resize_pre
            )
            dataset.prepare_data()
            dataset.setup("validate")

            dataloader = dataset.val_dataloader()

            x, _ = next(iter(dataloader))

            for transform_name in ["none", "translate", "scale"]:
                # TODO: can't fill with min value for CIFAR and ImageNet
                if transform_name == "translate":
                    transform = v2.RandomAffine(
                        degrees=0, translate=(0.5, 0.5), fill=x.min().item()
                    )
                elif transform_name == "scale":
                    transform = v2.RandomAffine(degrees=0, scale=(0.25, 2), fill=x.min().item())
                else:

                    def transform(x):
                        return x

                x_transformed = torch.stack([transform(_x) for _x in x]).to(self.device)

                forward_out = self(x_transformed, None)

                vis_outputs = visualize_model_output(
                    self,
                    x_transformed,
                    forward_out,
                    n_to_plot=4,
                )

                # TODO: predict foveation path

                if dataset_name == self.trainer.datamodule.dataset_name:
                    suffix = "/ID"
                else:
                    suffix = "/OOD"

                if transform_name == "none":
                    suffix += " (NT)"
                elif transform_name == "translate":
                    suffix += " (T)"
                elif transform_name == "scale":
                    suffix += " (S)"

                self.logger.log_image(
                    key="Real images" + suffix,
                    images=[vis_outputs["real_images"]],
                )
                self.logger.log_image(
                    key="Foveation Visualizations" + suffix,
                    images=vis_outputs["fov_vis"],
                )
                self.logger.log_image(
                    key="Image Reconstructions" + suffix,
                    images=vis_outputs["reconstructed_images"],
                )
                self.logger.log_image(
                    key="Real Patches" + suffix, images=[vis_outputs["real_patches_grid"]]
                )
                self.logger.log_image(
                    key="Reconstructed Patches" + suffix,
                    images=[vis_outputs["reconstructed_patches_grid"]],
                )
                self.logger.log_image(
                    key="Absolute Latent Traversal" + suffix,
                    images=[vis_outputs["abs_latent_traversal_grid"]],
                )
                self.logger.log_image(
                    key="Latent Traversal Around Z" + suffix,
                    images=[vis_outputs["around_latent_traversal_grid"]],
                )

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

        # grad_norm = self.clip_gradients(
        #     opt, gradient_clip_val=self.grad_clip, gradient_clip_algorithm="norm"
        # )

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
        g = super().optimizer_step(
            epoch,
            batch_idx,
            optimizer,
            optimizer_idx,
            optimizer_closure,
            on_tpu=on_tpu,
            using_lbfgs=using_lbfgs,
        )
        grad_norm, param_norms = self._get_grad_norm()

        self.log(
            "max_grad_norm_clipped",
            max(param_norms) if len(param_norms) > 0 else -1,
            on_epoch=True,
            on_step=False,
            logger=True,
            prog_bar=True,
            reduce_fx=torch.max,
            sync_dist=True,
        )
        return g

    # def backward(self, loss, optimizer, optimizer_idx, *args: Any, **kwargs: Any) -> None:
    #     return super().backward(loss, optimizer, optimizer_idx, *args, **kwargs)

    def on_after_backward(self) -> None:
        # only update if the grad norm is below a specific threshold
        grad_norm, param_norms = self._get_grad_norm()
        skipped_update = 0.0
        if self.grad_skip_threshold > 0 and grad_norm > self.grad_skip_threshold:
            skipped_update = 1.0
            for p in self.parameters():
                if p.grad is not None:
                    p.grad = None

        self.log(
            "max_grad_norm_unclipped",
            max(param_norms) if len(param_norms) > 0 else -1,
            on_epoch=True,
            on_step=False,
            logger=True,
            prog_bar=True,
            reduce_fx=torch.max,
            sync_dist=True,
        )

        self.log(
            "n_skipped_grad",
            skipped_update,
            on_epoch=True,
            on_step=False,
            logger=True,
            # prog_bar=True,
            reduce_fx=torch.sum,
            sync_dist=True,
        )

        return super().on_after_backward()

    def _get_grad_norm(self):
        total_norm = 0
        parameters = [p for p in self.parameters() if p.grad is not None and p.requires_grad]
        parameter_norms = []
        for p in parameters:
            param_norm = p.grad.detach().data.norm(2)
            parameter_norms.append(param_norm.item())
            total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5
        return total_norm, parameter_norms

    def on_train_epoch_end(self) -> None:
        k = super().on_train_epoch_end()
        gc.collect()

        # delete stale W&B files (otherwise they clog up the tempdir)
        if self.trainer.global_rank == 0:
            # delete all pngs in MEDIA_TMP older than 30 min ago
            for f in os.listdir(MEDIA_TMP.name):
                if f.endswith(".png"):
                    f = os.path.join(MEDIA_TMP, f)
                    if time.time() - os.path.getmtime(f) > 30 * 60:
                        os.remove(f)

        return k

    #     # skip updates with nans
    #     if True:
    #         # the closure (which includes the `training_step`) will be executed by optimizer.step
    #         optimizer.step(closure=optimizer_closure)
    #     else:
    #         # call the closure by itself to run `training_step` + `backward` without
    #         # an optimizer step
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
