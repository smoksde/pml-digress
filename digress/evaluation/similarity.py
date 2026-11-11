import torch
from rdkit import DataStructs

FP_RADIUS, FP_BITS = 2, 2048
THRESHOLDS = (0.8, 0.9, 1.0)


def nearest_similarity(fps, pool, exclude_self=False):
    out = []
    for k, fp in enumerate(fps):
        sims = torch.tensor(DataStructs.BulkTanimotoSimilarity(fp, pool))
        if exclude_self:
            sims[k] = -1.0
        out.append(sims.max().item())
    return torch.tensor(out)


def similarity_summary(sims):
    out = {"median": sims.median().item(), "q90": sims.quantile(0.9).item()}
    for t in THRESHOLDS:
        n = int((sims >= t - 1e-9).sum())
        out[t] = (n, n / len(sims))
    return out
