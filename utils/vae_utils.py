from typing import Optional
import torch


# @torch.jit.script
def gaussian_likelihood(x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, batch_reduce_fn="mean"):
    try:
        # scale = torch.exp(torch.ones_like(x_hat) * logscale)
        # mean = x_hat
        if mu.ndim == 1:
            mu = mu.unsqueeze(0).expand_as(x)
        if logvar.ndim == 1:
            logvar = logvar.unsqueeze(0).expand_as(x)

        assert (
            x.shape == mu.shape == logvar.shape
        ), f"Shapes of x, mu and logvar must match. Got {x.shape}, {mu.shape}, {logvar.shape}"
        std = torch.exp(0.5 * logvar)
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
    std: Optional[torch.Tensor] = None,
    logvar: Optional[torch.Tensor] = None,
    mu_prior=0.0,
    std_prior=1.0,
    batch_reduce_fn="mean",
    # free_bits: Optional[float] = None,
):
    if std is None and logvar is None:
        raise ValueError("Either std or logvar must be provided")
    elif std is not None and logvar is not None:
        raise ValueError("Only one of std or logvar must be provided")

    try:
        if std is None:
            std = torch.exp(0.5 * logvar)

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
        kl = kl.sum(-1)

    except ValueError as e:
        print(e)
        return torch.nan

    if batch_reduce_fn == "mean":
        kl = kl.mean()
    elif batch_reduce_fn == "sum":
        kl = kl.sum()
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
    return kl.clamp(min=free_bits) # .mean(0)


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
        return torch.empty_like(mu).fill_(torch.nan)

