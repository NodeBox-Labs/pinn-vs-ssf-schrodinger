"""Adaptive loss weighting schemes."""
import torch


class GradNorm:
    """Balance loss terms so their gradient norms w.r.t. network params match."""
    def __init__(self, n_terms=3, alpha=0.9, lr=0.01, init=None):
        self.alpha = alpha
        self.w = torch.tensor(init or [1.0] * n_terms)
        self._initial_gn = None
        self.lr = lr

    def update(self, losses, model):
        gns = []
        for L in losses:
            params = [p for p in model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(L, params, retain_graph=True,
                                        allow_unused=True)
            gn = torch.sqrt(sum((g ** 2).sum() for g in grads if g is not None))
            gns.append(gn)
        gns = torch.stack(gns)
        if self._initial_gn is None:
            self._initial_gn = gns.detach().clone()
        target = (gns / self._initial_gn) ** self.alpha
        self.w = self.w * (target.detach() / gns.detach()).clamp(0.5, 2.0)
        self.w = self.w / self.w.sum() * len(losses)
        return self.w


class SelfAdaptive:
    """Uncertainty weighting: L = Σ ½/s_i² L_i + log s_i."""
    def __init__(self, n_terms=3):
        self.log_s = torch.nn.Parameter(torch.zeros(n_terms))

    def apply(self, losses):
        s = torch.exp(self.log_s)
        return sum(0.5 / s[i] ** 2 * losses[i] + self.log_s[i]
                   for i in range(len(losses)))
