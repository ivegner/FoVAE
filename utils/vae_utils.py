from typing import Optional
import torch

# @torch.jit.script
def gaussian_likelihood(x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, reduce="mean"):
    try:
        # scale = torch.exp(torch.ones_like(x_hat) * logscale)
        # mean = x_hat
        assert x.shape == mu.shape == logvar.shape, (
            f"Shapes of x, mu and logvar must match. Got {x.shape}, {mu.shape}, {logvar.shape}"
        )
        std = torch.exp(0.5 * logvar)
        dist = torch.distributions.Normal(mu, std)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)

        # s = log_pxz.sum(dim=1)
        s = log_pxz.mean(dim=1) # CHANGED!
        # s = -torch.nn.functional.mse_loss(x, mu)

    except ValueError as e:
        print(e)
        return torch.nan

    if reduce == "mean":
        s = s.mean()
    elif reduce == "sum":
        s = s.sum()
    elif reduce == "none":
        pass
    else:
        raise ValueError(f"Unknown reduce value: {reduce}")
    return s

# @torch.jit.script
def gaussian_kl_divergence(
    mu: torch.Tensor,
    std: Optional[torch.Tensor] = None,
    logvar: Optional[torch.Tensor] = None,
    mu_prior=0.0,
    std_prior=1.0,
    reduce="mean",
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

    if reduce == "mean":
        kl = kl.mean()
    elif reduce == "sum":
        kl = kl.sum()
    elif reduce == "none":
        pass
    else:
        raise ValueError(f"Unknown reduce value: {reduce}")
    return kl
