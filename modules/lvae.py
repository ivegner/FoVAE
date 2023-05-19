from copy import deepcopy
import numpy as np
import torch
from torch import nn
from typing import List, Optional, Tuple

from utils.vae_utils import reparam_sample, norm_raw_logstd
from .transformers import VisionTransformer


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class FFBlock(nn.Module):
    def __init__(self, in_dim, out_dim, batch_norm=False, weight_norm=False):
        super().__init__()
        self.do_batch_norm = batch_norm
        if batch_norm:# and i != len(hidden_ff_out_dims) - 1:
            self.bn = nn.BatchNorm1d(in_dim)

        self.gelu = nn.LeakyReLU()
        if weight_norm:
            self.lin = nn.utils.weight_norm(nn.Linear(in_dim, out_dim))
            # https://github.com/pytorch/pytorch/issues/28594#issuecomment-1149882811
            self.lin.weight = self.lin.weight_v.detach()
        else:
            self.lin = nn.Linear(in_dim, out_dim)
        # torch.nn.utils.parametrizations.spectral_norm
        # nn.utils.weight_norm, nn.BatchNorm1d(last_out_dim)

    def forward(self, x):
        if self.do_batch_norm:
            x = self.bn(x)
        x = self.gelu(x)
        x = self.lin(x)
        return x

class FFNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_ff_out_dims=None, batch_norm=False, weight_norm=False):
        super().__init__()
        # assert not (batch_norm and weight_norm), "Cannot have both batch and weight norm"
        self.out_dim = out_dim

        if not hidden_ff_out_dims:
            hidden_ff_out_dims = []

        # if no hidden dims provided, will contain just out_dim
        hidden_ff_out_dims = [*hidden_ff_out_dims, out_dim]

        stack = []
        last_out_dim = in_dim
        for i, nn_out_dim in enumerate(hidden_ff_out_dims):
            stack.append(FFBlock(last_out_dim, nn_out_dim, batch_norm=batch_norm, weight_norm=weight_norm))
            last_out_dim = nn_out_dim

        self.encoder = nn.Sequential(*stack)

    def forward(self, x):
        return self.encoder(x)


class Ladder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        layer_out_dims: List[int],
        layer_hidden_dims: Optional[List[List[int]]] = None,
        batch_norm: bool = False,
        weight_norm: bool = False,
    ):
        super().__init__()
        self.layer_out_dims = layer_out_dims

        layer_out_dims = [in_dim, *layer_out_dims]

        self.layers = nn.ModuleList(
            [
                FFNet(
                    layer_out_dims[i],
                    layer_out_dims[i + 1],
                    hidden_ff_out_dims=layer_hidden_dims[i] if layer_hidden_dims else None,
                    batch_norm=batch_norm,
                    weight_norm=weight_norm
                )
                for i in range(len(layer_out_dims) - 1)
            ]
        )

    def forward(self, x):
        ladder_outputs = []
        x = x.view(x.size(0), -1)
        _x = x
        for layer in self.layers:
            _x = layer(_x)
            ladder_outputs.append(_x)
        return ladder_outputs, x


class LadderVAE(nn.Module):
    def __init__(
        self,
        in_dim: int,
        ladder_dims: List[int],
        z_dims: List[int],
        inference_hidden_dims: Optional[List[List[int]]] = None,
        generative_hidden_dims: Optional[List[List[int]]] = None,
        batch_norm: bool = False,
        weight_norm: bool = False,
    ):
        super().__init__()

        n_vae_layers = len(ladder_dims)

        assert n_vae_layers == len(z_dims), "All LadderVAE spec parameters must have same length"

        if inference_hidden_dims is not None:
            assert n_vae_layers == len(
                inference_hidden_dims
            ), "Must have same number of inference layer specs as ladder layers"

        if generative_hidden_dims is not None:
            assert n_vae_layers == len(
                generative_hidden_dims
            ), "Must have same number of generative layer specs as ladder layers"

        self.inference_layers = nn.ModuleList(
            [
                FFNet(
                    ladder_dims[i],
                    z_dims[i] * 2,
                    hidden_ff_out_dims=inference_hidden_dims[i] if inference_hidden_dims else None,
                    batch_norm=batch_norm,
                    weight_norm=weight_norm
                )
                for i in range(n_vae_layers)
            ]
        )

        _z_dims = [in_dim, *z_dims]
        self.generative_layers = nn.ModuleList(
            [
                FFNet(
                    _z_dims[i + 1],
                    _z_dims[i] * 2,
                    hidden_ff_out_dims=generative_hidden_dims[i]
                    if generative_hidden_dims
                    else None,
                    batch_norm=batch_norm,
                    weight_norm=weight_norm
                )
                for i in range(n_vae_layers)
            ]
        )

        assert len(self.inference_layers) == len(
            self.generative_layers
        ), "Inference and generative layers should have same length"

        self.n_vae_layers = n_vae_layers
        self.ladder_dims = ladder_dims
        self.z_dims = z_dims

        # # value based on Theis et al. 2016 “A Note on the Evaluation of Generative Models.”
        # # uniform noise with std of 1/12, scaled from being appropriate for input [0,255] to [-1,1]
        # self.patch_noise_std = nn.Parameter(
        #     torch.tensor([np.sqrt(1 / 12 / 127.5)]), requires_grad=False
        # )

        self.inf_logstd_norm = "explin"
        self.gen_logstd_norm = "bounded"
        self.gen_std_bound_min = 0.001
        self.gen_std_bound_max = 1.0

    def forward(
        self,
        ladder_outputs: Tuple[List[torch.Tensor], torch.Tensor],
        top_gen_prior_mu_std: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        ladder_outs_by_layer, x = ladder_outputs
        assert (
            len(ladder_outs_by_layer) == self.n_vae_layers
        ), "Ladder outputs should have same length as number of layers"

        # inference
        mu_stds_inf = []
        for ladder_x, layer in zip(ladder_outs_by_layer, self.inference_layers):
            distribution = layer(ladder_x)
            z_dim = int(distribution.size(1) / 2)
            assert z_dim == (distribution.size(1) / 2), "Inference latent dimension should be even"
            mu, logstd = distribution[:, :z_dim], distribution[:, z_dim:]
            _, std = norm_raw_logstd(logstd, self.inf_logstd_norm)
            mu_stds_inf.append((mu, std))

        # generative
        gen = self.generate(
            inference_mu_stds=mu_stds_inf, top_gen_prior_mu_std=top_gen_prior_mu_std
        )

        return dict(
            mu_stds_inference=mu_stds_inf,  # len(n_vae_layers)
            mu_stds_gen_prior=gen["mu_stds_gen_prior"],  # len(n_vae_layers+1)
            mu_stds_gen=gen["mu_stds_gen"],  # len(n_vae_layers+1)
            sample_zs=gen["sample_zs"],  # len(n_vae_layers+1)
        )

    def generate(
        self,
        inference_mu_stds: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        top_z: Optional[torch.Tensor] = None,
        top_gen_prior_mu_std: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        assert (
            inference_mu_stds is not None or top_z is not None or top_gen_prior_mu_std is not None
        ), "Must provide inference mu+std or top z or top generative prior mu+std"

        assert (top_z is None) or (
            top_gen_prior_mu_std is None
        ), "If providing top z, top generative prior parameters are meaningless"

        if top_z is not None:
            mu_stds_gen_prior = [(None, None)]
            mu_stds_gen = [(None, None)]
            sample_zs = []
        elif top_gen_prior_mu_std is not None:
            mu_stds_gen_prior = [top_gen_prior_mu_std]
            mu_stds_gen = []
            sample_zs = []
        else:
            mu_stds_gen_prior = [(None, None)]
            mu_stds_gen = []
            sample_zs = []

        for i, layer in reversed(list(enumerate(self.generative_layers))):
            is_top_layer = i == len(self.generative_layers) - 1
            is_bottom_layer = i == 0

            if is_top_layer and top_z is not None:
                z = top_z
            else:
                # get prior mu, std
                mu_gen_prior, std_gen_prior = mu_stds_gen_prior[0]

                # get inference mu, std
                mu_inf, std_inf = (
                    inference_mu_stds[i] if inference_mu_stds is not None else (None, None)
                )

                if is_top_layer and mu_gen_prior is None and std_gen_prior is None:
                    # if no prior, use inference
                    mu, std = mu_inf, std_inf
                elif mu_inf is None and std_inf is None:
                    # if no inference (i.e. generation), use prior
                    mu, std = mu_gen_prior, std_gen_prior
                else:
                    # combine inference and generative parameters by inverse variance weighting
                    # TODO: there has to be a way to do this without exponentiation
                    # also TODO: explore parametric combination methods
                    # (e.g. concat + linear transform)

                    # TODO! REWRITE TO USE STDS AND NORMS

                    # _, _std_inf = decode_raw_logstd_explin(std_inf)
                    # _, _std_gen_prior = decode_raw_logstd_explin(std_gen_prior)

                    _var_inf = std_inf.pow(2)
                    _var_gen = std_gen_prior.pow(2)

                    var = 1 / (1 / _var_inf + 1 / _var_gen)
                    mu = var * (mu_inf / _var_inf + mu_gen_prior / _var_gen)
                    std = torch.sqrt(var)

                mu_stds_gen.insert(0, (mu, std))
                # generate sample
                z = reparam_sample(mu, std)
            sample_zs.insert(0, z)

            # create next prior mu, std from sample
            distribution = layer(z)
            z_dim = int(distribution.size(1) / 2)
            assert z_dim == (distribution.size(1) / 2), "Generative latent dimension should be even"
            next_mu_gen_prior, next_raw_logstd_gen_prior = (
                distribution[:, :z_dim],
                distribution[:, z_dim:],
            )
            # if is_bottom_layer:
            #     _, next_std_gen_prior = norm_raw_logstd(
            #         next_raw_logstd_gen_prior, "bounded", self.patch_noise_std, 1.0
            #     )
            # else:
            if self.gen_logstd_norm == "constant":
                next_std_gen_prior = torch.ones_like(next_raw_logstd_gen_prior)
            elif self.gen_logstd_norm == "explin":
                _, next_std_gen_prior = norm_raw_logstd(next_raw_logstd_gen_prior, "explin")
            elif self.gen_logstd_norm == "bounded":
                _, next_std_gen_prior = norm_raw_logstd(
                    next_raw_logstd_gen_prior,
                    "bounded",
                    self.gen_std_bound_min,
                    self.gen_std_bound_max,
                )

            mu_stds_gen_prior.insert(0, (next_mu_gen_prior, next_std_gen_prior))

        # sample patch
        patch_mu_gen_prior, patch_std_gen_prior = mu_stds_gen_prior[0]
        mu_stds_gen.insert(
            0, (patch_mu_gen_prior, patch_std_gen_prior)
        )  # image generated from prior
        patch = reparam_sample(patch_mu_gen_prior, patch_std_gen_prior)
        sample_zs.insert(0, patch)

        assert (
            len(mu_stds_gen_prior)
            == len(mu_stds_gen)
            == len(sample_zs)
            == len(self.generative_layers) + 1
        )

        return dict(
            mu_stds_gen_prior=mu_stds_gen_prior,  # len(n_vae_layers+1)
            mu_stds_gen=mu_stds_gen,  # len(n_vae_layers+1)
            sample_zs=sample_zs,  # len(n_vae_layers+1)
        )


class NextPatchPredictor(nn.Module):
    def __init__(
        self,
        ladder_vae: LadderVAE,
        z_dims: List[int],
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_heads: int = 1,
        num_layers: int = 3,
        do_lateral_connections: bool = True,
        do_sigmoid_next_location: bool = False,
        do_flag_last_step: bool = False,
    ):
        super().__init__()

        # reuse generative layers from ladder vae
        self.ladder_vae = deepcopy(ladder_vae)
        self.ladder_vae.generative_layers = ladder_vae.generative_layers

        self.z_dims = z_dims
        self.do_lateral_connections = do_lateral_connections
        self.do_sigmoid_next_location = do_sigmoid_next_location

        self.top_z_predictor = VisionTransformer(
            input_dim=z_dims[-1] + 2,  # 2 for concatenated next position
            output_dim=z_dims[-1] * 2,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=0,
            do_flag_last_step=do_flag_last_step,
        )

        self.next_location_predictor = VisionTransformer(
            input_dim=z_dims[-1],
            output_dim=2 * 2,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=0,
            do_flag_last_step=do_flag_last_step,
        )

        self.loc_std_min = 0.01

    def forward(
        self,
        patch_step_zs: List[List[torch.Tensor]],
        curr_patch_ladder_outputs: List[torch.Tensor],
        forced_next_location: torch.Tensor = None,
        randomize_next_location: torch.Tensor = None,
        mask_to_last_step: bool = False,
    ):
        # patch_step_zs: n_steps_so_far x (n_levels from low to high) x (b, dim)
        # highest-level z is the last element of the list

        n_steps = len(patch_step_zs)
        n_levels = len(patch_step_zs[0])
        # top_zs = [patch_step_zs[i][-1] for i in range(n_steps)]
        b = patch_step_zs[0][0].size(0)
        device = patch_step_zs[0][0].device

        if mask_to_last_step:
            mask = torch.zeros(b, n_steps, device=device)
            mask[:, -1] = 1
        else:
            mask = None

        if forced_next_location is not None:
            next_pos, next_pos_mu, next_pos_std = forced_next_location, None, None
        else:
            next_pos, next_pos_mu, next_pos_std = self.pred_next_location(patch_step_zs, mask=mask)

        # randomize next location for those that are masked to true in randomize_next_location
        if randomize_next_location is not None:
            next_pos_rand = self._get_random_foveation_pos(b, device=device)
            next_pos = torch.where(randomize_next_location[:, None], next_pos_rand, next_pos)

        next_patch_gen_dict = self.generate_next_patch_zs(
            patch_step_zs,
            next_pos,
            curr_patch_ladder_outputs=curr_patch_ladder_outputs,
            mask=mask,
        )

        return dict(
            generation=next_patch_gen_dict,
            position=dict(
                next_pos=next_pos,
                next_pos_mu=next_pos_mu,
                next_pos_std=next_pos_std,
            ),
        )

    def pred_next_location(self, patch_step_zs: List[List[torch.Tensor]], mask=None):
        Z_LEVEL_TO_PRED_LOC = -1  # TODO: make this a param, and maybe multiple levels

        prev_top_zs = self._get_zs_from_level(patch_step_zs, Z_LEVEL_TO_PRED_LOC)
        pred = self.next_location_predictor(prev_top_zs, mask=mask)
        next_loc_mu, next_loc_raw_logstd = pred[:, :2], pred[:, 2:]

        # TODO: make params for this
        _, next_loc_std = norm_raw_logstd(next_loc_raw_logstd, "bounded", self.loc_std_min, 1.0)

        next_loc = reparam_sample(next_loc_mu, next_loc_std)

        if self.do_sigmoid_next_location:
            next_loc = nn.functional.sigmoid(next_loc) * 2 - 1
        else:
            next_loc = torch.clamp(next_loc, -1, 1)
        return (
            next_loc,
            next_loc_mu,
            next_loc_std,
        )

    def generate_next_patch_zs(
        self,
        patch_step_zs: List[List[torch.Tensor]],
        next_loc: torch.Tensor,
        curr_patch_ladder_outputs: Optional[List[torch.Tensor]] = None,
        mask=None,
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

        next_top_z_pred = self.top_z_predictor(prev_top_zs_with_pos, mask=mask)
        next_top_z_mu, next_top_z_raw_logstd = (
            next_top_z_pred[:, : self.z_dims[-1]],
            next_top_z_pred[:, self.z_dims[-1] :],
        )

        # TODO: parameterize instead of reusing ladder_vae's implicitly
        _, next_top_z_std = norm_raw_logstd(
            next_top_z_raw_logstd,
            self.ladder_vae.gen_logstd_norm,
            self.ladder_vae.gen_std_bound_min,
            self.ladder_vae.gen_std_bound_max,
        )

        # next_top_z = reparam_sample(next_top_z_mu, next_top_z_logstd)

        if self.do_lateral_connections and curr_patch_ladder_outputs is not None:
            # run inference from ladder outputs, combine with top-down z prediction
            next_patch_gen_dict = self.ladder_vae(
                curr_patch_ladder_outputs,
                top_gen_prior_mu_std=(next_top_z_mu, next_top_z_std),
            )
        else:
            # just do top-down generation
            next_patch_gen_dict = self.ladder_vae.generate(
                top_gen_prior_mu_std=(next_top_z_mu, next_top_z_std),
            )

        return next_patch_gen_dict

    def _get_zs_from_level(self, patch_step_zs: List[List[torch.Tensor]], level: int):
        s = torch.stack([zs[level] for zs in patch_step_zs], dim=0)
        s = s.transpose(0, 1)  # (b, n_steps, dim)
        return s

    def _get_random_foveation_pos(self, batch_size: int, device: torch.device = None):
        return torch.rand((batch_size, 2), device=device) * 2 - 1
