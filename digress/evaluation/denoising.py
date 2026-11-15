import torch

from digress.config import DEVICE, DIFFUSION_STEPS
from digress.data.chem import ATOM_LIST, NUM_ATOM_TYPES, NUM_BOND_TYPES
from digress.data.dataset import get_dataset
from digress.report.tables import Table
from digress.training.loss import predict

T_EDGES = [round(f * DIFFUSION_STEPS) for f in (0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)]
BOND_LABELS = ("none", "single", "double", "triple")


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
    data, out = get_dataset(), {}
    for lo, hi in zip(T_EDGES[:-1], T_EDGES[1:]):
        cm_atoms = torch.zeros(NUM_ATOM_TYPES, NUM_ATOM_TYPES, dtype=torch.long)
        cm_bonds = torch.zeros(NUM_BOND_TYPES, NUM_BOND_TYPES, dtype=torch.long)
        for start in range(0, len(idx), batch_size):
            batch = data.batch(idx[start : start + batch_size])
            t = torch.randint(lo + 1, hi + 1, (len(batch[0]),), device=DEVICE)
            atom_logits, bond_logits, atoms0, bonds0, mask0, edge_mask = predict(run, batch, t)
            cm_atoms += _confusion(atom_logits, atoms0, mask0, NUM_ATOM_TYPES)
            cm_bonds += _confusion(bond_logits, bonds0, edge_mask, NUM_BOND_TYPES)
        out[f"t{lo + 1}-{hi}"] = {"atoms": class_metrics(cm_atoms), "bonds": class_metrics(cm_bonds)}
    return out


def denoising_table(results):
    rows = [(k, f"{r['bonds']['accuracy']:.2f}", f"{r['bonds']['balanced_accuracy']:.2f}",
             *(f"{v:.2f}" for v in r["bonds"]["recall"][1:]), f"{r['atoms']['balanced_accuracy']:.2f}")
            for k, r in results.items()]
    return Table("denoising", "Clean-graph prediction on the test split by noise level.",
                 ("t", "Bond acc", "Bond bAcc", *(f"R {b}" for b in BOND_LABELS[1:]), "Atom bAcc"), rows,
                 ("acc: accuracy; bAcc: balanced accuracy (mean recall over classes); R: recall of a bond class. "
                  "Recall per atom type is shown in the recall figure; all metrics are in test_results.json.",))


def _per_entry_ce(marginal, alpha_bar):
    m = marginal.double().cpu()
    p_z_x = alpha_bar * torch.eye(len(m), dtype=torch.double) + (1 - alpha_bar) * m[None]
    post = m[:, None] * p_z_x
    post = post / post.sum(0, keepdim=True)
    return -(m[:, None] * p_z_x * post.clamp(min=1e-12).log()).sum().item()


def per_entry_reference(run, edge_weight, ts):
    def loss(a):
        return _per_entry_ce(run.atom_diffusion.m, a) + edge_weight * _per_entry_ce(run.bond_diffusion.m, a)

    return {f"t{t}": loss(run.atom_diffusion.alpha_bars[t - 1].item()) for t in ts}, loss(0.0)
