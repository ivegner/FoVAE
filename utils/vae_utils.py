from typing import Callable, Literal, Optional, Union
import torch


# @torch.jit.script
def gaussian_likelihood(
    x: torch.Tensor,
    mu: torch.Tensor,
    std: torch.Tensor,
    batch_reduce_fn="mean",
    # logstd_norm_method: Literal["none", "explin", "bounded"] = "none",
    # logstd_norm_bound_min: Optional[float] = None,
    # logstd_norm_bound_max: Optional[float] = None,
):
    try:
        # scale = torch.exp(torch.ones_like(x_hat) * logscale)
        # mean = x_hat
        if mu.ndim == 1:
            mu = mu.unsqueeze(0).expand_as(x)
        if std.ndim == 1:
            std = std.unsqueeze(0).expand_as(x)

        assert (
            x.shape == mu.shape == std.shape
        ), f"Shapes of x, mu and std must match. Got {x.shape}, {mu.shape}, {std.shape}"
        # logstd, std = norm_raw_logstd(
        #     raw_logstd,
        #     logstd_norm_method,
        #     norm_std_bound_min=logstd_norm_bound_min,
        #     norm_std_bound_max=logstd_norm_bound_max,
        # )

        dist = torch.distributions.Normal(mu, std)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)

        # s = log_pxz.sum(dim=1)
        s = log_pxz.mean(dim=1)  # CHANGED!
        # s = -torch.nn.functional.mse_loss(x, mu)

    except ValueError as e:
        print(e)
        return torch.nan

    if batch_reduce_fn == "mean":
        s = s.mean()
    elif batch_reduce_fn == "sum":
        s = s.sum()
    elif batch_reduce_fn == "none":
        pass
    else:
        raise ValueError(f"Unknown batch_reduce_fn value: {batch_reduce_fn}")
    return s


# @torch.jit.script
def gaussian_kl_divergence(
    mu: torch.Tensor,
    std: torch.Tensor,
    mu_prior: Optional[Union[float, torch.Tensor]] = None,
    std_prior: Optional[float] = None,
    batch_reduce_fn="mean",
    # logstd_norm_method: Literal["none", "explin", "bounded"] = "none",
    # logstd_norm_bound_min: Optional[float] = None,
    # logstd_norm_bound_max: Optional[float] = None,
):
    # if std is None and logstd is None:
    #     raise ValueError("Either std or logstd must be provided")
    # elif std is not None and logstd is not None:
    #     raise ValueError("Only one of std or logstd must be provided")
    # do_norm_prior = True

    if mu_prior is None and std_prior is None:
        mu_prior = torch.zeros_like(mu)
        std_prior = torch.ones_like(std)
        # do_norm_prior = False
    elif mu_prior is not None and std_prior is not None:
        if (
            isinstance(mu_prior, (float, int))
            or isinstance(mu_prior, torch.Tensor)
            and mu_prior.ndim == 0
        ):
            mu_prior = torch.ones_like(mu) * mu_prior
        if (
            isinstance(std_prior, (float, int))
            or isinstance(std_prior, torch.Tensor)
            and std_prior.ndim == 0
        ):
            std_prior = torch.ones_like(std) * std_prior
    else:
        raise ValueError(
            "If either mu_prior or raw_std_prior is provided, both must be provided"
        )
    try:
        # q_logstd, q_std = norm_raw_logstd(
        #     std,
        #     logstd_norm_method,
        #     norm_std_bound_min=logstd_norm_bound_min,
        #     norm_std_bound_max=logstd_norm_bound_max,
        # )
        # p_logstd, p_std = norm_raw_logstd(
        #     std_prior,
        #     logstd_norm_method=logstd_norm_method if do_norm_prior else "none",
        #     norm_std_bound_min=logstd_norm_bound_min,
        #     norm_std_bound_max=logstd_norm_bound_max,
        # )

        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(
            torch.ones_like(mu) * mu_prior, torch.ones_like(std) * std_prior
        )
        q = torch.distributions.Normal(mu, std)

        # # 2. get the probabilities from the equation
        # # log(q(z|x)) - log(p(z))
        # log_qzx = q.log_prob(z)
        # log_pz = p.log_prob(z)
        # kl = (log_qzx - log_pz)
        kl = torch.distributions.kl_divergence(q, p)

        # kl = kl.sum(-1)

    except ValueError as e:
        print(e)
        return torch.nan

    if batch_reduce_fn == "mean":
        kl = kl.mean(dim=0)
    elif batch_reduce_fn == "sum":
        kl = kl.sum(dim=0)
    elif batch_reduce_fn == "none":
        pass
    else:
        raise ValueError(f"Unknown batch_reduce_fn value: {batch_reduce_fn}")

    return kl


def free_bits_kl(
    kl: torch.Tensor,
    free_bits: float,
    # batch_average: Optional[bool] = False,
    # eps: Optional[float] = 1e-6,
) -> torch.Tensor:
    """Computes free-bits version of KL divergence.

    Takes in the KL with shape (batch size,), returns the scalar KL with
    free bits (for optimization) which is the average free-bits KL
    for the current batch
    If batch_average is False (default), the free bits are
    per batch element. Otherwise, the free bits are
    assigned on average to the whole batch. In both cases, the batch
    average is returned, so it's simply a matter of doing mean(clamp(KL))
    or clamp(mean(KL)).

    Adapted from https://github.com/addtt/boiler-pytorch.

    Args:
        kl (torch.Tensor): The KL with shape (batch size,)
        free_bits (float): Free bits
        batch_average (bool, optional)):
            If True, free bits are computed for average batch KL. If False, computed for KL
            of each element and then averaged across batch.

    Returns:
        The KL with free bits
    """
    # assert kl.dim() == 1
    # if free_bits < eps:
    #     return kl.mean(0)
    # if batch_average:
    #     return kl.mean(0).clamp(min=free_bits)
    return kl.clamp(min=free_bits)  # .mean(0)


@torch.jit.script
def _reparam_sample(mu, std):
    # std = torch.exp(0.5 * logstd)
    eps = torch.empty_like(mu).normal_(0.0, 1.0)
    return mu + std * eps


def reparam_sample(
    mu,
    std
    # raw_logstd,
    # logstd_norm_method: Literal["none", "explin", "bounded"] = "none",
    # logstd_norm_bound_min: Optional[float] = None,
    # logstd_norm_bound_max: Optional[float] = None,
):
    """Reparameterization trick for sampling from a Gaussian distribution.

    Args:
        mu (torch.Tensor): Mean of the Gaussian distribution
        std (torch.Tensor): std of the Gaussian distribution

    Returns:
        torch.Tensor: Sample from the Gaussian distribution
    """
    # _, std = norm_raw_logstd(
    #     raw_logstd,
    #     logstd_norm_method,
    #     norm_std_bound_min=logstd_norm_bound_min,
    #     norm_std_bound_max=logstd_norm_bound_max,
    # )

    i = 0
    while i < 20:
        # randn_like sometimes produces NaNs for unknown reasons
        # maybe see: https://github.com/pytorch/pytorch/issues/46155
        # so we try again if that happens
        s = _reparam_sample(mu, std)
        if not torch.isnan(s).any():
            return s
        # print(f"Could not sample without NaNs (try {i})")
        i += 1
    else:
        print("Could not sample from N(0, 1) without NaNs after 20 tries")
        print("mu:", mu.max(), mu.min())
        # print("raw_logstd:", raw_logstd.max(), raw_logstd.min())
        print("std:", std.max(), std.min())
        return torch.empty_like(mu).fill_(torch.nan)


def norm_raw_logstd(
    logstd,
    logstd_norm_method: Literal["none", "explin", "bounded"] = "none",
    norm_std_bound_min: Optional[float] = None,
    norm_std_bound_max: Optional[float] = None,
):
    """Normalizes the raw logstd.

    Args:
        logstd (torch.Tensor): Raw logstd
        logstd_norm_method (str, optional): Normalization method for raw_logstd. Defaults to "none".
        See Dehaene and Brossard 2021, “Re-Parameterizing VAEs for Stability.”
        norm_std_bound_min (float, optional): Minimum bound for std derived from "bounded" methods.
            Defaults to None for no bound.
        norm_std_bound_max (float, optional): Maximum bound for std derived from "bounded" methods.
            Defaults to None for no bound.

    Returns:
        torch.Tensor: Normalized logstd
    """
    if logstd_norm_method == "none":
        return decode_raw_logstd_naive(logstd)
    elif logstd_norm_method == "explin":
        return decode_raw_logstd_explin(logstd)
    elif logstd_norm_method == "bounded":
        if norm_std_bound_min is None and norm_std_bound_max is None:
            raise ValueError(
                "At least one of logstd_norm_bound_min or logstd_norm_bound_max must be "
                "provided if logstd_norm_method is 'bounded'"
            )
        elif norm_std_bound_min is not None and norm_std_bound_max is not None:
            return decode_raw_logstd_bounded(logstd, norm_std_bound_min, norm_std_bound_max)
        elif norm_std_bound_min is not None:
            return decode_raw_logstd_downbounded(logstd, min_std_value=norm_std_bound_min)
        elif norm_std_bound_max is not None:
            return decode_raw_logstd_upbounded(logstd, max_std_value=norm_std_bound_max)


# def gaussian_kl_divergence_naive(
#     p_mu: torch.Tensor, p_std: torch.Tensor, q_mu: torch.Tensor, q_std: torch.Tensor
# ):
#     """Pytorch naive implementation of the KL divergence between two Gaussians."""
#     return _base_gaussian_kl(p_mu, p_std, torch.log(p_std), q_mu, q_std, torch.log(q_std))


# def gaussian_kl_divergence_exp(
#     p_mu: torch.Tensor, p_logstd: torch.Tensor, q_mu: torch.Tensor, q_logstd: torch.Tensor
# ):
#     """More numerically stable naive KL implementation of KL divergence between two Gaussians."""
#     return _base_gaussian_kl(
#         p_mu, torch.exp(p_logstd), p_logstd, q_mu, torch.exp(q_logstd), q_logstd
#     )


# def gaussian_kl_divergence_explin(
#     p_mu: torch.Tensor, p_logstd_raw: torch.Tensor, q_mu: torch.Tensor, q_logstd_raw: torch.Tensor
# ):
#     """KL divergence in which exponentiation is replaced by linear function for high values of logstd."""
#     p_logstd, p_std = decode_raw_logstd_explin(p_logstd_raw)
#     q_logstd, q_std = decode_raw_logstd_explin(q_logstd_raw)

#     return _base_gaussian_kl(p_mu, p_std, p_logstd, q_mu, q_std, q_logstd)


@torch.jit.script
def _base_gaussian_kl(
    p_mu: torch.Tensor,
    p_std: torch.Tensor,
    p_logstd: torch.Tensor,
    q_mu: torch.Tensor,
    q_std: torch.Tensor,
    q_logstd: torch.Tensor,
):
    """Base implementation for Gaussian KL that can be used for many implementations."""
    return q_logstd - p_logstd + (p_std.pow(2) + (p_mu - q_mu).pow(2)) / (2 * q_std.pow(2)) - 0.5


@torch.jit.script
def decode_raw_logstd_naive(p: torch.Tensor):
    """Decodes raw logstd from a network output into logstd and std."""
    return p, torch.exp(p)


@torch.jit.script
def decode_raw_logstd_explin(p: torch.Tensor):
    """Decodes raw logstd from a network output into logstd and std.

    Uses a linear function for high values of logstd.
    """
    p_clipmin = torch.clip(p, min=0)
    p_clipmax = torch.clip(p, max=0)

    mask = p_clipmin > 0

    logstd = torch.where(mask, torch.log1p(p_clipmin), p_clipmax)
    std = torch.where(mask,  1 + p_clipmin, torch.exp(p_clipmax))
    return logstd, std


@torch.jit.script
def decode_raw_logstd_upbounded(p: torch.Tensor, max_std_value: float = 1.0):
    """Decodes raw logstd from a network output into logstd and std.

    Caps the maximum std to max_value.
    """
    p_clipmin = torch.clip(p, min=0)
    p_clipmax = torch.clip(p, max=0)

    mask = p_clipmin > 0

    _g = torch.tensor(max_std_value/2, dtype=p.dtype, device=p.device)
    _log_g = torch.log(_g)
    _exp_min = torch.exp(-p_clipmin)

    logstd = torch.where(mask, torch.log(2 - _exp_min) + _log_g, p_clipmax + _log_g)
    std = torch.where(mask, _g * (2 - _exp_min), _g * torch.exp(p_clipmax))
    return logstd, std


@torch.jit.script
def decode_raw_logstd_downbounded(p: torch.Tensor, min_std_value: float = 0.0):
    """Decodes raw logstd from a network output into logstd and std.

    Caps the minimum std to min_value. Useful for decoding with minimum noise.
    """
    std = min_std_value + torch.exp(p)
    logstd = torch.log(std)
    return logstd, std


@torch.jit.script
def decode_raw_logstd_bounded(
    p: torch.Tensor,
    min_std_value: float = 0.0,
    max_std_value: float = 1.0,
    # scaling_function: Callable = torch.sigmoid,
):
    """Decodes raw logstd from a network output into logstd and std.

    Caps the minimum std to min_value and maximum std to max_value,
    using sigmoid to normalize.
    Useful for decoding with minimum noise and a maximum cap.
    """
    std = min_std_value + (max_std_value - min_std_value) * torch.sigmoid(p)
    logstd = torch.log(std)
    return logstd, std
