"""
changes: t_edges(T) replaces the const. , confusion matrices sizing updated to use info and bond labels include aromatic if used (mainly for MOSES).
"""
import torch

from digress.config import DEVICE
from digress.report.tables import Table
from digress.training.loss import predict

T_FRACTIONS = (0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
BOND_LABELS = ("none", "single", "double", "triple")  # DatasetInfo.bond_labels can add "aromatic"


def t_edges(T):
    return [round(f * T) for f in T_FRACTIONS]


def _confusion(logits, target, mask, k):
    return torch.bincount(target[mask] * k + logits.argmax(-1)[mask], minlength=k * k).view(k, k).cpu()


def class_metrics(cm):
    cm = cm.float()
    recall, precision = cm.diag() / cm.sum(1).clamp(min=1), cm.diag() / cm.sum(0).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
    present = cm.sum(1) > 0
    return {"accuracy": (cm.diag().sum() / cm.sum()).item(), "balanced_accuracy": recall[present].mean().item(),
            "macro_f1": f1[present].mean().item(), "recall": recall.tolist()}


def denoising_metrics(run, idx, batch_size=500):
    data, info, out = run.data, run.info, {}
    ka, kb = info.num_atom_types, info.num_bond_types
    edges = t_edges(run.T)
    for lo, hi in zip(edges[:-1], edges[1:]):
        cm_atoms = torch.zeros(ka, ka, dtype=torch.long)
        cm_bonds = torch.zeros(kb, kb, dtype=torch.long)
        for start in range(0, len(idx), batch_size):
            batch = data.batch(idx[start : start + batch_size])
            t = torch.randint(lo + 1, hi + 1, (len(batch[0]),), device=DEVICE)
            atom_logits, bond_logits, atoms0, bonds0, mask0, edge_mask = predict(run, batch, t)
            cm_atoms += _confusion(atom_logits, atoms0, mask0, ka)
            cm_bonds += _confusion(bond_logits, bonds0, edge_mask, kb)
        out[f"t{lo + 1}-{hi}"] = {"atoms": class_metrics(cm_atoms), "bonds": class_metrics(cm_bonds)}
    return out


def denoising_table(results, bond_labels=BOND_LABELS):
    n_bonds = len(next(iter(results.values()))["bonds"]["recall"])
    labels = tuple(bond_labels) + tuple(f"class {k}" for k in range(len(bond_labels), n_bonds))
    rows = [(k, f"{r['bonds']['accuracy']:.2f}", f"{r['bonds']['balanced_accuracy']:.2f}",
             *(f"{v:.2f}" for v in r["bonds"]["recall"][1:]), f"{r['atoms']['balanced_accuracy']:.2f}")
            for k, r in results.items()]
    return Table("denoising", "Clean-graph prediction on the test split by noise level.",
                 ("t", "Bond acc", "Bond bAcc", *(f"R {b}" for b in labels[1:n_bonds]), "Atom bAcc"), rows,
                 ("acc: accuracy; bAcc: balanced accuracy (mean recall over classes); R: recall of a bond class. "
                  "Recall per atom type is shown in the recall figure; all metrics are in test_results.json.",))


def _per_entry_ce(marginal, alpha_bar):
    m = marginal.double().cpu()
    p_z_x = alpha_bar * torch.eye(len(m), dtype=torch.double) + (1 - alpha_bar) * m[None]
    post = m[:, None] * p_z_x
    post = post / post.sum(0, keepdim=True)
    return -(m[:, None] * p_z_x * post.clamp(min=1e-12).log()).sum().item()


def per_entry_reference(run, edge_weight, ts):
    """
    Loss of the best predictor that only sees each entry's own noisy class. Assumes the data marginals as both
    class prior and noise limit, i.e. exact for the marginal transition
    """
    atom_m, bond_m = run.marginals["atom_marginal"], run.marginals["bond_marginal"]

    def loss(a):
        return _per_entry_ce(atom_m, a) + edge_weight * _per_entry_ce(bond_m, a)

    return {f"t{t}": loss(run.atom_diffusion.alpha_bars[t - 1].item()) for t in ts}, loss(0.0)
