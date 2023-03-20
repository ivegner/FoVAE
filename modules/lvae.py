import torch
from torch import nn
from typing import List, Optional, Tuple

from utils.vae_utils import reparam_sample
from .transformers import VisionTransformer


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
    def __init__(
        self,
        in_dim: int,
        layer_out_dims: List[int],
        layer_hidden_dims: Optional[List[List[int]]] = None,
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
        inference_hidden_dims: Optional[List[List[int]]] = None,
        generative_hidden_dims: Optional[List[List[int]]] = None,
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
    def __init__(
        self,
        ladder_vae: LadderVAE,
        z_dims: List[int],
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_heads: int = 1,
        num_layers: int = 3,
        do_random_foveation: bool = False,
    ):
        super().__init__()

        self.ladder_vae = ladder_vae
        self.z_dims = z_dims
        self.do_random_foveation = do_random_foveation

        self.top_z_predictor = VisionTransformer(
            input_dim=z_dims[-1] + 2,  # 2 for concatenated next position
            output_dim=z_dims[-1] * 2,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=0,
        )

        self.next_location_predictor = VisionTransformer(
            input_dim=z_dims[-1],
            output_dim=2 * 2,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
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

