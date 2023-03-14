import gc
from copy import deepcopy
from typing import Any, Iterable, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
from torch import nn, optim
from timeit import default_timer as timer


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


@torch.jit.script
def _reparam_sample(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.empty_like(mu).normal_(0.0, 1.0)
    return mu + std * eps


def reparam_sample(mu, logvar):
    i = 0
    while i < 20:
        # randn_like sometimes produces NaNs for unknown reasons
        # maybe see: https://github.com/pytorch/pytorch/issues/46155
        # so we try again if that happens
        s = _reparam_sample(mu, logvar)
        if not torch.isnan(s).any():
            return s
        # print(f"Could not sample without NaNs (try {i})")
        i += 1
    else:
        print("Could not sample from N(0, 1) without NaNs after 20 tries")
        print("mu:", mu.max(), mu.min())
        print("logvar:", logvar.max(), logvar.min())
        return torch.nan


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class FFNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_ff_out_dims=None):
        super().__init__()
        self.out_dim = out_dim

        if not hidden_ff_out_dims:
            hidden_ff_out_dims = []

        # if no hidden dims provided, will contain just out_dim
        hidden_ff_out_dims = [*hidden_ff_out_dims, out_dim]

        stack = []
        last_out_dim = in_dim
        for nn_out_dim in hidden_ff_out_dims:
            stack.extend(
                [
                    nn.GELU(),
                    # torch.nn.utils.parametrizations.spectral_norm(
                    nn.Linear(last_out_dim, nn_out_dim)
                    # ),
                ]
            )  # nn.utils.weight_norm, nn.BatchNorm1d(last_out_dim)
            last_out_dim = nn_out_dim

        self.encoder = nn.Sequential(*stack)

    def forward(self, x):
        return self.encoder(x)


class Ladder(nn.Module):
    def __init__(self, in_dim: int, layer_out_dims: List[int], layer_hidden_dims: List[List[int]]):
        super().__init__()
        self.layer_out_dims = layer_out_dims

        layer_out_dims = [in_dim, *layer_out_dims]

        self.layers = nn.ModuleList(
            [
                FFNet(
                    layer_out_dims[i],
                    layer_out_dims[i + 1],
                    hidden_ff_out_dims=layer_hidden_dims[i],
                )
                for i in range(len(layer_out_dims) - 1)
            ]
        )

    def forward(self, x):
        ladder_outputs = []
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
            ladder_outputs.append(x)
        return ladder_outputs


class LadderVAE(nn.Module):
    def __init__(
        self,
        in_dim: int,
        ladder_dims: List[int],
        z_dims: List[int],
        inference_hidden_dims: List[List[int]],
        generative_hidden_dims: List[List[int]],
    ):
        super().__init__()

        n_vae_layers = len(ladder_dims)

        assert (
            n_vae_layers == len(z_dims) == len(inference_hidden_dims) == len(generative_hidden_dims)
        ), "All LadderVAE spec parameters must have same length"

        self.inference_layers = nn.ModuleList(
            [
                FFNet(ladder_dims[i], z_dims[i] * 2, hidden_ff_out_dims=inference_hidden_dims[i])
                for i in range(n_vae_layers)
            ]
        )

        _z_dims = [in_dim, *z_dims]
        self.generative_layers = nn.ModuleList(
            [
                FFNet(_z_dims[i + 1], _z_dims[i] * 2, hidden_ff_out_dims=generative_hidden_dims[i])
                for i in range(n_vae_layers)
            ]
        )

        assert len(self.inference_layers) == len(
            self.generative_layers
        ), "Inference and generative layers should have same length"

        self.n_vae_layers = n_vae_layers
        self.ladder_dims = ladder_dims
        self.z_dims = z_dims

    def forward(self, ladder_outputs: List[torch.Tensor]):
        assert (
            len(ladder_outputs) == self.n_vae_layers
        ), "Ladder outputs should have same length as number of layers"

        # inference
        mu_logvars_inf = []
        for ladder_x, layer in zip(ladder_outputs, self.inference_layers):
            distribution = layer(ladder_x)
            z_dim = int(distribution.size(1) / 2)
            assert z_dim == (distribution.size(1) / 2), "Inference latent dimension should be even"
            mu, logvar = distribution[:, :z_dim], distribution[:, z_dim:]
            mu_logvars_inf.append((mu, logvar))

        # generative
        gen = self.generate(inference_mu_logvars=mu_logvars_inf)

        return dict(
            mu_logvars_inference=mu_logvars_inf,  # len(n_vae_layers)
            mu_logvars_gen_prior=gen["mu_logvars_gen_prior"],  # len(n_vae_layers+1)
            mu_logvars_gen=gen["mu_logvars_gen"],  # len(n_vae_layers+1)
            sample_zs=gen["sample_zs"],  # len(n_vae_layers+1)
        )

    def generate(
        self,
        inference_mu_logvars: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        top_z: Optional[torch.Tensor] = None,
        top_gen_prior_mu_logvar: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        assert (
            inference_mu_logvars is not None
            or top_z is not None
            or top_gen_prior_mu_logvar is not None
        ), "Must provide inference mu+logvar or top z or top generative prior mu+logvar"

        assert (top_z is None) or (
            top_gen_prior_mu_logvar is None
        ), "If providing top z, top generative prior parameters are meaningless"

        if top_z is not None:
            mu_logvars_gen_prior = [(None, None)]
            mu_logvars_gen = [(None, None)]
            sample_zs = []
        elif top_gen_prior_mu_logvar is not None:
            mu_logvars_gen_prior = [top_gen_prior_mu_logvar]
            mu_logvars_gen = []
            sample_zs = []
        else:
            mu_logvars_gen_prior = [(None, None)]
            mu_logvars_gen = []
            sample_zs = []

        for i, layer in reversed(list(enumerate(self.generative_layers))):
            is_top_layer = i == len(self.generative_layers) - 1

            if is_top_layer and top_z is not None:
                z = top_z
            else:
                # get prior mu, logvar
                mu_gen_prior, logvar_gen_prior = mu_logvars_gen_prior[0]

                # get inference mu, logvar
                mu_inf, logvar_inf = (
                    inference_mu_logvars[i] if inference_mu_logvars is not None else (None, None)
                )

                if is_top_layer and mu_gen_prior is None and logvar_gen_prior is None:
                    # if no prior, use inference
                    mu, logvar = mu_inf, logvar_inf
                elif mu_inf is None and logvar_inf is None:
                    # if no inference (i.e. generation), use prior
                    mu, logvar = mu_gen_prior, logvar_gen_prior
                else:
                    # combine inference and generative parameters by inverse variance weighting
                    # TODO: there has to be a way to do this without exponentiation
                    # also TODO: explore parametric combination methods
                    # (e.g. concat + linear transform)

                    _var_inf = torch.exp(logvar_inf)
                    _var_gen = torch.exp(logvar_gen_prior)

                    var = 1 / (1 / _var_inf + 1 / _var_gen)
                    mu = var * (mu_inf / _var_inf + mu_gen_prior / _var_gen)
                    logvar = torch.log(var)

                mu_logvars_gen.insert(0, (mu, logvar))
                # generate sample
                z = reparam_sample(mu, logvar)
            sample_zs.insert(0, z)

            # create next prior mu, logvar from sample
            distribution = layer(z)
            z_dim = int(distribution.size(1) / 2)
            assert z_dim == (distribution.size(1) / 2), "Generative latent dimension should be even"
            next_mu_gen_prior, next_logvar_gen_prior = (
                distribution[:, :z_dim],
                distribution[:, z_dim:],
            )
            mu_logvars_gen_prior.insert(0, (next_mu_gen_prior, next_logvar_gen_prior))

        # sample patch
        patch_mu_gen_prior, patch_logvar_gen_prior = mu_logvars_gen_prior[0]
        mu_logvars_gen.insert(
            0, (patch_mu_gen_prior, patch_logvar_gen_prior)
        )  # image generated from prior
        patch = reparam_sample(patch_mu_gen_prior, patch_logvar_gen_prior)
        sample_zs.insert(0, patch)

        assert (
            len(mu_logvars_gen_prior)
            == len(mu_logvars_gen)
            == len(sample_zs)
            == len(self.generative_layers) + 1
        )

        return dict(
            mu_logvars_gen_prior=mu_logvars_gen_prior,  # len(n_vae_layers+1)
            mu_logvars_gen=mu_logvars_gen,  # len(n_vae_layers+1)
            sample_zs=sample_zs,  # len(n_vae_layers+1)
        )


class NextPatchPredictor(nn.Module):
    def __init__(self, ladder_vae: LadderVAE, z_dims: List[int], do_random_foveation: bool = False):
        super().__init__()

        self.ladder_vae = ladder_vae
        self.z_dims = z_dims
        self.do_random_foveation = do_random_foveation

        self.top_z_predictor = VisionTransformer(
            input_dim=z_dims[-1] + 2,  # 2 for concatenated next position
            output_dim=z_dims[-1] * 2,
            embed_dim=256,  # TODO
            hidden_dim=512,  # TODO
            num_heads=1,  # TODO
            num_layers=3,  # TODO
            dropout=0,
        )

        self.next_location_predictor = VisionTransformer(
            input_dim=z_dims[-1],
            output_dim=2 * 2,
            embed_dim=256,  # TODO
            hidden_dim=512,  # TODO
            num_heads=1,  # TODO
            num_layers=3,  # TODO
            dropout=0,
        )

    def forward(
        self,
        patch_step_zs: List[List[torch.Tensor]],
        forced_next_location: torch.Tensor = None,
        randomize_next_location: bool = False,
    ):
        # patch_step_zs: n_steps_so_far x (n_levels from low to high) x (b, dim)
        # highest-level z is the last element of the list

        n_steps = len(patch_step_zs)
        # n_levels = len(patch_step_zs[0])
        top_zs = [patch_step_zs[i][-1] for i in range(n_steps)]
        b = top_zs[0].size(0)
        device = top_zs[0].device

        if forced_next_location is not None:
            next_pos, next_pos_mu, next_pos_logvar = forced_next_location, None, None
        elif self.do_random_foveation or randomize_next_location:
            next_pos, next_pos_mu, next_pos_logvar = (
                self._get_random_foveation_pos(b, device=device),
                None,
                None,
            )
        else:
            next_pos, next_pos_mu, next_pos_logvar = self.pred_next_location(patch_step_zs)

        next_patch_gen_dict = self.generate_next_patch_zs(patch_step_zs, next_pos)

        return dict(
            generation=next_patch_gen_dict,
            position=dict(
                next_pos=next_pos,
                next_pos_mu=next_pos_mu,
                next_pos_logvar=next_pos_logvar,
            ),
        )

    def pred_next_location(self, patch_step_zs: List[List[torch.Tensor]]):
        Z_LEVEL_TO_PRED_LOC = -1  # TODO: make this a param, and maybe multiple levels

        prev_top_zs = self._get_zs_from_level(patch_step_zs, Z_LEVEL_TO_PRED_LOC)
        pred = self.next_location_predictor(prev_top_zs)
        next_loc_mu, next_loc_logvar = pred[:, :2], pred[:, 2:]
        next_loc = reparam_sample(next_loc_mu, next_loc_logvar)

        return (
            torch.clamp(next_loc, -1, 1),
            next_loc_mu,
            next_loc_logvar,
        )

    def generate_next_patch_zs(
        self, patch_step_zs: List[List[torch.Tensor]], next_loc: torch.Tensor
    ):
        n_steps = len(patch_step_zs)
        # n_levels = len(patch_step_zs[0])
        b = patch_step_zs[0][0].size(0)
        assert next_loc.size() == torch.Size([b, 2]), "next_loc should be (b, 2)"

        Z_LEVEL_TO_PRED_PATCH = -1

        prev_top_zs = self._get_zs_from_level(patch_step_zs, Z_LEVEL_TO_PRED_PATCH)
        # concatenate next patch pos to each z, TODO: maybe add as extra token instead?
        prev_top_zs_with_pos = torch.cat(
            (prev_top_zs, next_loc.unsqueeze(1).repeat(1, n_steps, 1)),
            dim=2,
        )

        next_top_z_pred = self.top_z_predictor(prev_top_zs_with_pos)
        next_top_z_mu, next_top_z_logvar = (
            next_top_z_pred[:, : self.z_dims[-1]],
            next_top_z_pred[:, self.z_dims[-1] :],
        )
        # next_top_z = reparam_sample(next_top_z_mu, next_top_z_logvar)

        next_patch_gen_dict = self.ladder_vae.generate(
            top_gen_prior_mu_logvar=(next_top_z_mu, next_top_z_logvar)
        )

        return next_patch_gen_dict

    def _get_zs_from_level(self, patch_step_zs: List[List[torch.Tensor]], level: int):
        s = torch.stack([zs[level] for zs in patch_step_zs], dim=0)
        s = s.transpose(0, 1)  # (b, n_steps, dim)
        return s

    def _get_random_foveation_pos(self, batch_size: int, device: torch.device = None):
        return torch.rand((batch_size, 2), device=device) * 2 - 1


# def zip_reverse(*iterables):
#     return zip(*[reversed(g) for g in iterables])


# class UpBlock(nn.Module):
#     def __init__(self, from_sample_dim, to_z_dim, hidden_ff_out_dims=None):
#         super().__init__()
#         self.out_z_dim = to_z_dim

#         if not hidden_ff_out_dims:
#             hidden_ff_out_dims = []

#         # if no hidden dims provided, will contain just z dim (*2 bc mean and logvar)
#         hidden_ff_out_dims = [*hidden_ff_out_dims, to_z_dim * 2]

#         stack = []
#         last_out_dim = from_sample_dim
#         for nn_out_dim in hidden_ff_out_dims:
#             stack.extend([nn.GELU(), nn.Linear(last_out_dim, nn_out_dim)])
#             last_out_dim = nn_out_dim

#         self.encoder = nn.Sequential(View((-1, from_sample_dim)), *stack)

#     def forward(self, x):
#         distributions = self.encoder(x)
#         mu = distributions[:, : self.out_z_dim]
#         logvar = distributions[:, self.out_z_dim :]
#         return mu, logvar


# class DownBlock(nn.Module):
#     def __init__(self, from_dim, to_dim, hidden_ff_out_dims=None):
#         super().__init__()

#         if not hidden_ff_out_dims:
#             hidden_ff_out_dims = []

#         # if no hidden dims provided, will contain just to_dim
#         hidden_ff_out_dims = [*hidden_ff_out_dims, to_dim]

#         stack = []
#         last_out_dim = from_dim
#         for nn_out_dim in hidden_ff_out_dims:
#             stack.extend([nn.GELU(), nn.Linear(last_out_dim, nn_out_dim)])
#             last_out_dim = nn_out_dim

#         self.decoder = nn.Sequential(View((-1, from_dim)), *stack)

#     def forward(self, x):
#         return self.decoder(x)


class FoVAE(pl.LightningModule):
    def __init__(
        self,
        image_dim=28,
        fovea_radius=2,
        patch_dim=6,
        patch_channels=3,
        n_vae_levels=1,
        z_dim=10,
        n_steps: int = 1,
        foveation_padding: Union[Literal["max"], int] = "max",
        foveation_padding_mode: Literal["zeros", "replicate"] = "replicate",
        lr=1e-3,
        beta=1,
        # n_spectral_iter=1,
        # grad_clip=100,
        grad_skip_threshold=-1,
        # do_add_pos_encoding=True,
        do_z_pred_cond_from_top=True,
        do_use_beta_norm=True,
        do_random_foveation=False,
        do_image_reconstruction=True,
        do_next_patch_prediction=True,
        reconstruct_fovea_only=False
    ):
        super().__init__()

        self.image_dim = image_dim
        self.fovea_radius = fovea_radius
        self.patch_dim = patch_dim
        self.z_dim = z_dim

        self.n_vae_levels = n_vae_levels

        self.n_steps = n_steps
        # if do_add_pos_encoding:
        self.num_channels = patch_channels + 2
        self.lr = lr
        self.foveation_padding = foveation_padding
        self.foveation_padding_mode = foveation_padding_mode

        # left/right singular vectors used for SR
        self.n_spectral_power_iter = 1  # n_spectral_iter
        self.sr_u = {}
        self.sr_v = {}

        input_dim = self.patch_dim * self.patch_dim * self.num_channels

        if n_vae_levels == 1:
            VAE_LADDER_DIMS = [32]
            VAE_Z_DIMS = [z_dim]
            LADDER_HIDDEN_DIMS = [[512, 512]]
            LVAE_INF_HIDDEN_DIMS = [[256, 256]]
            LVAE_GEN_HIDDEN_DIMS = [[256, 256]]
        elif n_vae_levels == 2:
            VAE_LADDER_DIMS = [32, 32]
            VAE_Z_DIMS = [z_dim, z_dim // 2]
            LADDER_HIDDEN_DIMS = [[512, 512], [256, 256]]
            LVAE_INF_HIDDEN_DIMS = [[256, 256], [128, 128]]
            LVAE_GEN_HIDDEN_DIMS = [[256, 256], [128, 128]]

        self.ladder = Ladder(input_dim, VAE_LADDER_DIMS, LADDER_HIDDEN_DIMS)
        self.ladder_vae = LadderVAE(
            input_dim, VAE_LADDER_DIMS, VAE_Z_DIMS, LVAE_INF_HIDDEN_DIMS, LVAE_GEN_HIDDEN_DIMS
        )
        self.next_patch_predictor = NextPatchPredictor(
            ladder_vae=self.ladder_vae,
            z_dims=VAE_Z_DIMS,
            do_random_foveation=do_random_foveation,
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

        self._beta = beta

        self.grad_skip_threshold = grad_skip_threshold
        # self.do_add_pos_encoding = do_add_pos_encoding
        self.do_z_pred_cond_from_top = do_z_pred_cond_from_top

        if do_use_beta_norm:
            beta_vae = (beta * z_dim) / input_dim  # according to beta-vae paper
            print(
                f"Using normalized betas[1] value of {beta_vae:.6f} as beta, "
                f"calculated from unnormalized beta_vae {beta:.6f}"
            )
        else:
            beta_vae = beta

        self.betas = dict(
            curr_patch_recon=1,
            curr_patch_kl=beta_vae,
            next_patch_pos_kl=1,
            next_patch_recon=1,
            next_patch_kl=beta_vae,
            image_recon=20,
            spectral_norm=0,
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
        self.reconstruct_fovea_only = reconstruct_fovea_only

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

        x_full = self._add_pos_encodings_to_img_batch(x)

        DO_KL_ON_INPUT_LEVEL = False

        curr_patch_rec_total_loss = 0.0
        curr_patch_kl_div_total_loss = 0.0
        next_patch_pos_kl_div_total_loss = 0.0
        next_patch_rec_total_loss = 0.0
        next_patch_kl_div_total_loss = 0.0
        image_reconstruction_loss = 0.0
        curr_patch_kl_divs_by_layer, next_patch_rec_losses_by_layer, next_patch_kl_divs_by_layer = (
            [],
            [],
            [],
        )

        def memoized_patch_getter(x_full):
            _fov_memo = None

            def get_patch_from_pos(pos):
                # TODO: investigate why reshape vs. view is needed
                nonlocal _fov_memo
                patch, _fov_memo = self._foveate_to_loc(x_full, pos, _fov_memo=_fov_memo)
                patch = patch.reshape(b, -1)
                assert patch.shape == (b, self.num_channels * self.patch_dim * self.patch_dim)
                return patch

            return get_patch_from_pos

        get_patch_from_pos = memoized_patch_getter(x_full)

        initial_positions = torch.tensor([0.0, 0.0], device=x.device).unsqueeze(0).repeat(b, 1)
        patches = [get_patch_from_pos(initial_positions)]
        patch_positions = [initial_positions]

        real_patch_zs = []
        real_patch_dicts = []

        gen_patch_zs = []
        gen_patch_dicts = []

        for step in range(self.n_steps):
            curr_patch = patches[-1]
            curr_patch_dict = self._process_patch(curr_patch)
            # curr_patch_dict:
            #   mu_logvars_inference: list(n_vae_layers) of (mu, logvar) tuples, each (b, z_dim)
            #   mu_logvars_gen_prior: list(n_vae_layers+1) of (mu, logvar) tuples, each (b, z_dim)
            #   mu_logvars_gen: list(n_vae_layers+1) of (mu, logvar) tuples, each (b, z_dim)
            #   sample_zs: list(n_vae_layers+1) of (b, z_dim)
            # each list is in order from bottom to top.
            # gen lists and sample_zs have input-level at index 0
            assert torch.is_same_size(curr_patch, curr_patch_dict["sample_zs"][0])
            real_patch_zs.append(curr_patch_dict["sample_zs"])
            real_patch_dicts.append(curr_patch_dict)

            if self.do_next_patch_prediction:
                next_patch_dict = self._gen_next_patch(
                    real_patch_zs, randomize_next_location=self.do_random_foveation
                )
                # next_patch_dict:
                #   generation:
                #     mu_logvars_gen_prior: list(n_vae_layers+1) of (mu, logvar) tuples,
                #                                                   each (b, z_dim)
                #     mu_logvars_gen: list(n_vae_layers+1) of (mu, logvar) tuples, each (b, z_dim)
                #     sample_zs: list(n_vae_layers+1) of (b, z_dim)
                #   position:
                #     next_pos: (b, 2)
                #     next_pos_mu: (b, 2)
                #     next_pos_logvar: (b, 2)
                gen_patch_zs.append(next_patch_dict["generation"]["sample_zs"])
                gen_patch_dicts.append(next_patch_dict)

                next_pos = next_patch_dict["position"]["next_pos"]
            elif self.do_random_foveation:
                next_pos = self._get_random_foveation_pos(batch_size=b)
            else:
                raise ValueError("Must do either next patch prediction or random foveation")

            # foveate to next position
            next_patch = get_patch_from_pos(next_pos)
            assert torch.is_same_size(next_patch, curr_patch)
            patches.append(next_patch)
            patch_positions.append(next_pos)

            # calculate losses

            # calculate rec and kl losses for current patch
            _curr_patch_rec_loss = -1 * self._patch_likelihood(
                curr_patch,
                mu=curr_patch_dict["mu_logvars_gen"][0][0],
                logvar=curr_patch_dict["mu_logvars_gen"][0][1],
                fovea_only=self.reconstruct_fovea_only,
            )

            _curr_patch_kl_divs = []
            for i, (mu, logvar) in enumerate(curr_patch_dict["mu_logvars_gen"]):
                kl = gaussian_kl_divergence(mu=mu, logvar=logvar)
                if i == 0 and not DO_KL_ON_INPUT_LEVEL:
                    kl = torch.zeros_like(kl)
                _curr_patch_kl_divs.append(kl)

            # calculate kl divergence between predicted next patch pos and std-normal prior
            # only do kl divergence because
            # reconstruction of next_pos is captured in next_patch_rec_loss
            _next_patch_pos_kl_div = 0.0
            if not self.do_random_foveation:
                _next_patch_pos_kl_div = gaussian_kl_divergence(
                    mu=next_patch_dict["position"]["next_pos_mu"],
                    logvar=next_patch_dict["position"]["next_pos_logvar"],
                )

            # if any previous predicted patch, calculate loss between
            # current patch and previous predicted patch
            _next_patch_rec_losses, _next_patch_kl_divs = [], []
            if self.do_next_patch_prediction and len(gen_patch_zs) > 1:
                # -2 because -1 is the current step, and -2 is the previous step
                prev_gen_patch_dict = gen_patch_dicts[-2]
                _next_patch_rec_losses, _next_patch_kl_divs = [], []
                for i, (mu, logvar) in enumerate(
                    prev_gen_patch_dict["generation"]["mu_logvars_gen"]
                ):
                    if i == 0:
                        # input-level, compare against real patch
                        level_rec_loss = -1 * self._patch_likelihood(
                            curr_patch, mu=mu, logvar=logvar, fovea_only=self.reconstruct_fovea_only
                        )
                    else:
                        level_rec_loss = -1 * self._patch_likelihood(
                            curr_patch_dict["sample_zs"][i],
                            mu=mu,
                            logvar=logvar,
                            # fovea_only=self.reconstruct_fovea_only,
                        )
                    level_kl = gaussian_kl_divergence(
                        mu=mu,
                        logvar=logvar,
                    )
                    if i == 0 and not DO_KL_ON_INPUT_LEVEL:
                        level_kl = torch.zeros_like(level_kl)

                    _next_patch_rec_losses.append(level_rec_loss)
                    _next_patch_kl_divs.append(level_kl)

                next_patch_rec_losses_by_layer.append(torch.stack(_next_patch_rec_losses, dim=0))
                next_patch_kl_divs_by_layer.append(torch.stack(_next_patch_kl_divs, dim=0))

            # aggregate losses
            curr_patch_rec_total_loss += _curr_patch_rec_loss
            curr_patch_kl_divs_by_layer.append(torch.stack(_curr_patch_kl_divs, dim=0))
            next_patch_pos_kl_div_total_loss += _next_patch_pos_kl_div

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
                real_patch_zs, x_full, return_patches=False, fovea_only=self.reconstruct_fovea_only
            )
        else:
            image_reconstruction_loss = torch.tensor(0.0, device=self.device)

        # TODO: there's a memory leak somewhere, comes out during overfit_batches=1
        # Notes on memory leak:
        # https://github.com/Lightning-AI/lightning/issues/16876
        # https://github.com/pytorch/pytorch/issues/13246
        # https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/

        # aggregate losses across steps
        # mean over steps
        curr_patch_kl_divs_by_layer = torch.stack(curr_patch_kl_divs_by_layer, dim=0).mean(dim=0)
        if self.do_next_patch_prediction:
            next_patch_rec_losses_by_layer = torch.stack(
                next_patch_rec_losses_by_layer, dim=0
            ).mean(dim=0)
            next_patch_kl_divs_by_layer = torch.stack(next_patch_kl_divs_by_layer, dim=0).mean(
                dim=0
            )

        curr_patch_rec_total_loss = (
            self.betas["curr_patch_recon"] * curr_patch_rec_total_loss / self.n_steps
        )
        # sum over layers (already mean over steps)
        curr_patch_kl_div_total_loss = (
            self.betas["curr_patch_kl"] * curr_patch_kl_divs_by_layer.sum()
        )
        next_patch_pos_kl_div_total_loss = (
            self.betas["next_patch_pos_kl"] * next_patch_pos_kl_div_total_loss / self.n_steps
        )
        # sum over layers (already mean over steps)
        if self.do_next_patch_prediction:
            next_patch_rec_total_loss = (
                self.betas["next_patch_recon"] * next_patch_rec_losses_by_layer.sum()
            )
            next_patch_kl_div_total_loss = (
                self.betas["next_patch_kl"] * next_patch_kl_divs_by_layer.sum()
            )

        image_reconstruction_loss = self.betas["image_recon"] * image_reconstruction_loss
        spectral_norm = (
            self.betas["spectral_norm"] * self.spectral_norm_parallel()
            if self.betas["spectral_norm"] > 0
            else 0.0
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
                curr_patch_kl_divs_by_layer=curr_patch_kl_divs_by_layer,  # n_levels
                next_patch_rec_losses_by_layer=next_patch_rec_losses_by_layer,  # n_levels
                next_patch_kl_divs_by_layer=next_patch_kl_divs_by_layer,  # n_levels
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

    def _patch_likelihood(self, patch, mu, logvar, fovea_only=False):
        if fovea_only:
            return gaussian_likelihood(
                self._patch_to_fovea(patch),
                mu=self._patch_to_fovea(mu),
                logvar=self._patch_to_fovea(logvar),
            )
        else:
            return gaussian_likelihood(patch, mu=mu, logvar=logvar)

    def _get_random_foveation_pos(self, batch_size: int):
        return self.next_patch_predictor._get_random_foveation_pos(batch_size, device=self.device)

    def _reconstruct_image(self, sample_zs, image: Optional[torch.Tensor], return_patches=False, fovea_only=False):
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

        def memoized_patch_getter(image):
            _fov_memo = None

            def get_patch_from_pos(pos):
                nonlocal _fov_memo
                patch, _fov_memo = self._foveate_to_loc(image, pos, _fov_memo=_fov_memo)
                return patch

            return get_patch_from_pos

        # predict zs for each position
        image_recon_loss = None
        gen_zs = [*sample_zs]
        patches = []
        if image is not None:
            _memo_foveate_to_loc = memoized_patch_getter(image)
        # TODO: maybe reconstruct only some patches?

        for i, position in enumerate(positions):
            gen_dict = self._gen_next_patch(gen_zs, forced_next_location=position)
            gen_patch = gen_dict["generation"]["sample_zs"][0]
            gen_mu, gen_logvar = gen_dict["generation"]["mu_logvars_gen"][0]
            # gen_zs.append(gen_dict["generation"]["sample_zs"])
            gen_patch = gen_patch.view(b, self.num_channels, self.patch_dim, self.patch_dim)
            if image is not None:
                if image_recon_loss is None:
                    image_recon_loss = 0.0
                real_patch = _memo_foveate_to_loc(position)
                assert torch.is_same_size(gen_patch, real_patch)

                patch_recon_loss = -1 * self._patch_likelihood(
                    real_patch.view(b, -1), mu=gen_mu, logvar=gen_logvar, fovea_only=fovea_only
                )
                image_recon_loss += patch_recon_loss

            patches.append(self._patch_to_fovea(gen_patch) if fovea_only else gen_patch)

        if image_recon_loss is not None:
            image_recon_loss /= positions.size(0)

        if return_patches:
            return image_recon_loss, torch.stack(patches, dim=0).transpose(0, 1)
        else:
            return image_recon_loss, None

    def _foveate_to_loc(self, image: torch.Tensor, loc: torch.Tensor, _fov_memo: dict = None):
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

        # pad_h, pad_w = padded_image.shape[-2:]

        # move the gaussian filter params to the loc
        gaussian_filter_params = self._move_default_filter_params_to_loc(loc, (h, w), pad_offset)

        # foveate
        foveated_image, _fov_memo = fov_utils.apply_mean_foveation_pyramid(
            padded_image, gaussian_filter_params, memo=_fov_memo
        )

        _fov_memo["orig_image"] = image
        _fov_memo["padded_image"] = padded_image
        _fov_memo["pad_offset"] = pad_offset

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
                    f"New gaussian centers after move not close to loc: "
                    f"{new_mus.mean(1)[torch.argmax((new_mus.mean(1) - loc).sum(1), 0)]} "
                    f"vs {loc[torch.argmax((new_mus.mean(1) - loc).sum(1), 0)]}"
                )
            ring["mus"] = new_mus + pad_offset

        return gaussian_filter_params

    def _process_patch(self, x: torch.Tensor):
        ladder_outputs = self.ladder(x)
        patch_vae_dict = self.ladder_vae(ladder_outputs)
        return patch_vae_dict

    def _gen_next_patch(
        self,
        prev_zs: List[List[torch.Tensor]],
        forced_next_location: Optional[torch.Tensor] = None,
        randomize_next_location: bool = False,
    ):
        # prev_zs: list(n_steps_so_far) of lists
        #              (n_levels from lowest to highest) of tensors (b, dim)
        # next_patch_pos: Tensor (b, 2)
        # highest-level z is the last element of the list

        return self.next_patch_predictor(
            prev_zs,
            forced_next_location=forced_next_location,
            randomize_next_location=randomize_next_location,
        )

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
    #         zs, mus, logvars = self._encode_patch(patch_with_pos)

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

        fovea = p[:, :, ring_radius:-ring_radius, ring_radius:-ring_radius]
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
            for row in range(self.z_dim):
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

    def training_step(self, batch, batch_idx):
        x, y = batch
        forward_out = self.forward(x, y)
        # total_loss = forward_out["losses"].pop("total_loss")
        total_loss = forward_out["losses"]["total_loss"]
        # self.log("train_total_loss", total_loss, prog_bar=True)
        self.log_dict({"train_" + k: v for k, v in forward_out["losses"].items()})

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
            prog_bar=True,
            reduce_fx=torch.sum,
        )
        # self.log(grad_norm, skip_update, on_epoch=True, logger=True)

        return None if skip_update else total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        forward_out = self.forward(x, y)
        total_loss = forward_out["losses"]["total_loss"]
        self.log_dict({"val_" + k: v for k, v in forward_out["losses"].items()})

        # plot kl divergences by layer on the same plots
        curr_patch_kl_divs_by_layer = forward_out["losses_by_layer"]["curr_patch_kl_divs_by_layer"]
        _curr_kl_divs = {
            f"curr_patch_kl_l{i}": v for i, v in enumerate(curr_patch_kl_divs_by_layer)
        }
        self.log("val_curr_patch_kl_by_layer", _curr_kl_divs)
        if self.do_next_patch_prediction:
            next_patch_kl_divs_by_layer = forward_out["losses_by_layer"][
                "next_patch_kl_divs_by_layer"
            ]
            _next_kl_divs = {
                f"next_patch_kl_l{i}": v for i, v in enumerate(next_patch_kl_divs_by_layer)
            }
            self.log("val_next_patch_kl_by_layer", _next_kl_divs)

        step_sample_zs = forward_out["step_vars"]["real_patch_zs"]
        # step_z_recons = forward_out["step_vars"]["z_recons"]
        step_next_z_preds = forward_out["step_vars"]["gen_patch_zs"]
        patches = forward_out["step_vars"]["patches"]
        step_patch_positions = forward_out["step_vars"]["patch_positions"]

        # step_sample_zs: (n_steps, n_layers, batch_size, z_dim)
        assert (
            patches[0][0].size()
            == step_sample_zs[0][0][0].size()
            == torch.Size([self.num_channels * self.patch_dim * self.patch_dim])
        )
        assert step_sample_zs[0][1][0].size() == torch.Size([self.z_dim])

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
            # gaussian_filter_params = _recursive_to(
            #     self._move_default_filter_params_to_loc(loc, (h, w), pad_offset=None),
            #     "cpu",
            # )
            # plot_gaussian_foveation_parameters(
            #                     x[[3]].cpu(),
            #                     gaussian_filter_params,
            #                     axs=[ax1],
            #                     point_size=10,
            #                 )
            # fov = self._foveate_to_loc(self._add_pos_encodings_to_img_batch(x[[3]]), loc).cpu()
            # imshow_unnorm(fov[0,[0]], ax=ax2)

            # make figure with a column for each step and 3 rows:
            # 1 for image with foveation, one for patch, one for patch reconstruction

            figs = [plt.figure(figsize=(self.n_steps * 3, 12)) for _ in range(N_TO_PLOT)]
            axs = [f.subplots(4, self.n_steps) for f in figs]

            # plot foveations on images
            for step, img_step_batch in enumerate(real_images):
                # positions = (
                #     patches[step]
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
                step_patch_batch = remove_pos_channels_from_batch(
                    patches[step][:N_TO_PLOT].view(
                        -1, self.num_channels, self.patch_dim, self.patch_dim
                    )
                )
                for i in range(N_TO_PLOT):
                    imshow_unnorm(step_patch_batch[i].cpu(), ax=axs[i][1][step])
                    axs[i][1][step].set_title(f"Patch at step {step}", fontsize=8)

            # plot patch reconstructions
            for step in range(self.n_steps):
                step_patch_batch = remove_pos_channels_from_batch(
                    step_sample_zs[step][0][:N_TO_PLOT].view(
                        -1, self.num_channels, self.patch_dim, self.patch_dim
                    )
                )
                for i in range(N_TO_PLOT):
                    imshow_unnorm(step_patch_batch[i].cpu(), ax=axs[i][2][step])
                    axs[i][2][step].set_title(f"Patch reconstruction at step {step}", fontsize=8)

            # plot next patch predictions
            if self.do_next_patch_prediction:
                for step in range(self.n_steps):
                    pred_patches = step_next_z_preds[step][0][:N_TO_PLOT].view(
                        -1, self.num_channels, self.patch_dim, self.patch_dim
                    )
                    pred_pos = (
                        pred_patches[:, -2:].mean(dim=(2, 3)) / 2 + 0.5
                    ).cpu() * torch.tensor([h, w])
                    pred_patches = remove_pos_channels_from_batch(pred_patches)
                    for i in range(N_TO_PLOT):
                        ax = axs[i][3][step]
                        imshow_unnorm(pred_patches[i].cpu(), ax=ax)
                        ax.set_title(
                            f"Next patch pred. at step {step} - "
                            f"({pred_pos[i][0]:.1f}, {pred_pos[i][1]:.1f})",
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

            # if self.do_image_reconstruction:
            _, reconstructed_images = self._reconstruct_image(
                [[level[:N_TO_PLOT] for level in step] for step in step_sample_zs],
                image=None,
                return_patches=True,
            )
            reconstructed_images = reconstructed_images.cpu()

            for i in range(N_TO_PLOT):
                # ax = axs[i]
                # imshow_unnorm(patches[i].cpu(), ax=ax)
                # ax.set_title(
                #     f"Next patch pred. at step {step} - "
                #     f"({pred_pos[i][0]:.1f}, {pred_pos[i][1]:.1f})",
                #     fontsize=8,
                # )
                tensorboard.add_image(
                    f"Image Reconstructions {i}",
                    torchvision.utils.make_grid(
                        remove_pos_channels_from_batch(
                            self._patch_to_fovea(reconstructed_images[i])
                        )
                        / 2
                        + 0.5,
                        nrow=int(np.sqrt(len(reconstructed_images[i]))),
                        padding=1,
                    ),
                    global_step=self.global_step,
                )

            # step constant bc real images don't change
            tensorboard.add_images(
                "Real Patches",
                remove_pos_channels_from_batch(
                    patches[0][:32].view(-1, self.num_channels, self.patch_dim, self.patch_dim) / 2
                    + 0.5
                ).cpu(),
                global_step=0,
            )
            tensorboard.add_images(
                "Reconstructed Patches",
                remove_pos_channels_from_batch(
                    step_sample_zs[0][0][:32].view(
                        -1, self.num_channels, self.patch_dim, self.patch_dim
                    )
                    / 2
                    + 0.5
                ).cpu(),
                global_step=self.global_step,
            )

            def stack_traversal_output(g):
                # stack by interp image, then squeeze out the singular batch dimension and
                # index out the 2 position channels
                return [
                    remove_pos_channels_from_batch(torch.stack(dt).squeeze(1))
                    for dt in traversal_abs
                ]

            # img = self._add_pos_encodings_to_img_batch(x[[0]])
            # get top-level z of first step of first image of batch.
            z_level = -1
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
        super().optimizer_step(
            epoch,
            batch_idx,
            optimizer,
            optimizer_idx,
            optimizer_closure,
            on_tpu=on_tpu,
            using_lbfgs=using_lbfgs,
        )

        # def _optimizer_step(self, loss):
        #     opt = self.optimizers()
        #     opt.zero_grad()
        #     self.manual_backward(loss)

        # grad_norm = self.clip_gradients(opt, gradient_clip_val=self.grad_clip, gradient_clip_algorithm="norm")
        # grad_norm = _get_grad_norm()

        # only update if loss is not NaN and if the grad norm is below a specific threshold
        skipped_update = 1
        # if self.grad_skip_threshold == -1 or grad_norm < self.grad_skip_threshold:
        #     skipped_update = 0
        #     optimizer.step(closure=optimizer_closure)
        #     # TODO: EMA updating
        #     # TODO: factor out loss NaNs by what produced them (kl or reconstruction)
        #     # update_ema(vae, ema_vae, H.ema_rate)
        # else:
        #     # call the closure by itself to run `training_step` + `backward` without
        #     # an optimizer step
        #     optimizer_closure()

    # def backward(self, loss, optimizer, optimizer_idx, *args: Any, **kwargs: Any) -> None:
    #     return super().backward(loss, optimizer, optimizer_idx, *args, **kwargs)

    def on_after_backward(self) -> None:
        # only update if the grad norm is below a specific threshold
        grad_norm = self._get_grad_norm()
        skipped_update = 0.0
        if self.grad_skip_threshold > 0 and grad_norm > self.grad_skip_threshold:
            skipped_update = 1.0
            for p in self.parameters():
                if p.grad is not None:
                    p.grad = None

        self.log(
            "n_skipped_grad",
            skipped_update,
            on_epoch=True,
            on_step=False,
            logger=True,
            prog_bar=True,
            reduce_fx=torch.sum,
        )

        return super().on_after_backward()

    def _get_grad_norm(self):
        total_norm = 0
        parameters = [p for p in self.parameters() if p.grad is not None and p.requires_grad]
        for p in parameters:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5
        return total_norm

    def on_train_epoch_end(self) -> None:
        k = super().on_train_epoch_end()
        gc.collect()
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
