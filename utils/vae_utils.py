import torch

def gaussian_likelihood(x, x_hat, logscale):
    # scale = torch.exp(torch.ones_like(x_hat) * logscale)
    mean = x_hat
    dist = torch.distributions.Normal(mean, 1.0)

    # measure prob of seeing image under p(x|z)
    log_pxz = dist.log_prob(x)
    return log_pxz.sum(dim=1)

def gaussian_kl_divergence(mu, std, mu_prior=0.0, std_prior=1.0):
    # --------------------------
    # Monte carlo KL divergence
    # --------------------------
    # 1. define the first two probabilities (in this case Normal for both)
    p = torch.distributions.Normal(torch.ones_like(mu) * mu_prior, torch.ones_like(std) * std_prior)
    q = torch.distributions.Normal(mu, std)

    # # 2. get the probabilities from the equation
    # # log(q(z|x)) - log(p(z))
    # log_qzx = q.log_prob(z)
    # log_pz = p.log_prob(z)
    # kl = (log_qzx - log_pz)
    kl = torch.distributions.kl_divergence(q, p)
    kl = kl.sum(-1)
    return kl
