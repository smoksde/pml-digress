import torch
import torch.nn.functional as F

from digress.config import COSINE_SCHEDULE_OFFSET
from digress.data.graphs import edge_mask_from_nodes


def cosine_alpha_schedule(T, s=COSINE_SCHEDULE_OFFSET):
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T + s) / (1 + s)) * torch.pi / 2) ** 2
    alpha_bars = f / f[0]
    return (alpha_bars[1:] / alpha_bars[:-1]).clamp(1e-4, 0.9999).float()


class DiscreteDiffusion:
    def __init__(self, marginal, num_timesteps):
        self.m = torch.as_tensor(marginal, dtype=torch.float32)
        self.K = len(self.m)
        self.alphas = cosine_alpha_schedule(num_timesteps)
        self.alpha_bars = torch.cumprod(self.alphas, 0)

    def q_sample(self, x0, t):
        shape = [x0.shape[0]] + [1] * (x0.dim() - 1)
        stay = self.alpha_bars.to(x0.device)[t - 1].view(shape).expand(x0.shape)
        mask = torch.bernoulli(1 - stay).bool()
        resampled = torch.multinomial(self.m.to(x0.device), x0.numel(), replacement=True).view(x0.shape)
        return torch.where(mask, resampled, x0)

    def posterior_probs(self, xt, pred_x0_probs, t):
        m, eye = self.m.to(xt.device), torch.eye(self.K, device=xt.device)
        a_t = self.alphas[t - 1].item()
        a_prev = self.alpha_bars[t - 2].item() if t > 1 else 1.0
        Q_t = a_t * eye + (1 - a_t) * m
        Qbar_prev = a_prev * eye + (1 - a_prev) * m
        p = Q_t[:, xt].movedim(0, -1) * (pred_x0_probs @ Qbar_prev)
        return p / p.sum(-1, keepdim=True).clamp(min=1e-12)

    def reverse_step(self, xt, logits, t):
        probs = self.posterior_probs(xt, torch.softmax(logits, -1), t)
        return torch.distributions.Categorical(probs=probs).sample()


def noise_graph(X, E, mask, t, atom_diffusion, bond_diffusion):
    mask = mask.bool()
    edge_mask = edge_mask_from_nodes(mask)
    t = torch.as_tensor(t, device=X.device).long().expand(X.shape[0])

    atoms_t = atom_diffusion.q_sample(X.argmax(-1), t)
    bonds_t = bond_diffusion.q_sample(E.argmax(-1), t)
    X_t = F.one_hot(atoms_t, X.shape[-1]).float() * mask.unsqueeze(-1)
    E_t = F.one_hot(bonds_t, E.shape[-1]).float() * edge_mask.unsqueeze(-1)
    return X_t, E_t
