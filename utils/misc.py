import torch

def recursive_to(x, *args, **kwargs):
    if isinstance(x, torch.Tensor):
        return x.to(*args, **kwargs)
    elif isinstance(x, dict):
        return {k: recursive_to(v, *args, **kwargs) for k, v in x.items()}
    elif isinstance(x, list):
        return [recursive_to(v, *args, **kwargs) for v in x]
    else:
        return x

def recursive_detach(g):
    if isinstance(g, dict):
        return {k: recursive_detach(v) for k, v in g.items()}
    elif isinstance(g, (list, tuple)):
        return [recursive_detach(v) for v in g]
    elif g is None:
        return None
    else:
        return g.detach()
