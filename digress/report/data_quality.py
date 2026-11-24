from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDetermineBonds

from digress.config import DATA_DIR, DIFFUSION_STEPS, MAX_ATOMS, RANDOM_SEED
from digress.data.chem import ATOM_LIST, NUM_ATOM_TYPES, NUM_BOND_TYPES, flat_smiles, mol_to_graph
from digress.data.info import QM9
from digress.data.graphs import edge_vector_to_adj
from digress.data.marginals import bootstrap_stats
from digress.diffusion.discrete import DiscreteDiffusion, noise_graph
from digress.evaluation.denoising import BOND_LABELS
from digress.evaluation.metrics import graph_histograms
from digress.report.molecules import _layout, draw_molecule, show
from digress.report.style import FULL_WIDTH, MUTED, figure, save
from digress.report.tables import Table, fmt_n, fmt_pct, write_tables

GDB_24 = 23
CHARGE_SAMPLE_EVERY = 13
REBUILD_OUTCOMES = (("same", "neutral; same bonds as stored graph"), ("different", "neutral; different bonds than stored"),
                    ("charged", "still charged (genuine)"), ("failed", "bond perception failed"))


def _share_rows(counts, total):
    return [(label, fmt_n(int(c)), fmt_pct(c / total)) for label, c in counts]


def _charged_atoms(mol):
    return [f"{a.GetSymbol()}{'+' if a.GetFormalCharge() > 0 else '-'}" for a in mol.GetAtoms() if a.GetFormalCharge()]


def mol_from_geometry(z, pos):
    mol = Chem.RWMol()
    for atomic_number in z.tolist():
        mol.AddAtom(Chem.Atom(int(atomic_number)))
    conformer = Chem.Conformer(len(z))
    for i, p in enumerate(pos.tolist()):
        conformer.SetAtomPosition(i, p)
    mol.AddConformer(conformer)
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=0)
        return Chem.RemoveHs(mol)
    except Exception:
        return None


def neutralized(mol):
    mol = Chem.RWMol(mol)
    for atom in mol.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(False)
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def _graph(mol):
    try:
        atoms, mask, adj = mol_to_graph(mol, QM9)
        return atoms, adj, mask
    except Exception:
        return None


def _rebuild_outcome(stored, rebuilt):
    if rebuilt is None:
        return "failed"
    if _charged_atoms(rebuilt):
        return "charged"
    stripped = neutralized(stored)
    return "same" if stripped is not None and flat_smiles(stripped) == flat_smiles(rebuilt) else "different"


def charge_analysis(every=CHARGE_SAMPLE_EVERY, n_examples=6):
    from torch_geometric.datasets import QM9

    RDLogger.DisableLog("rdApp.*")
    ds = QM9(root=DATA_DIR)
    mols = {i: Chem.MolFromSmiles(s) for i, s in enumerate(ds._data.smiles)}
    charged = [i for i, mol in mols.items() if mol is not None and _charged_atoms(mol)]
    patterns = Counter(" ".join(sorted(_charged_atoms(mols[i]))) for i in charged)
    net = sum(Chem.GetFormalCharge(mols[i]) != 0 for i in charged)
    sample = set(charged[::every])
    outcomes, examples = Counter(), []
    for i in dict.fromkeys([GDB_24, *charged[::every]]):
        rebuilt = mol_from_geometry(ds[i].z, ds[i].pos)
        if i in sample:
            outcomes[_rebuild_outcome(mols[i], rebuilt)] += 1
        pair = (_graph(mols[i]), _graph(rebuilt)) if rebuilt is not None else (None,)
        if len(examples) < n_examples and None not in pair and not _charged_atoms(rebuilt):
            examples.append((pair, Chem.MolToSmiles(mols[i], isomericSmiles=False), flat_smiles(rebuilt)))
    n, top = len(charged), patterns.most_common(6)
    pattern_table = Table(
        "charge_patterns", "Formal-charge patterns in PyG's QM9 SMILES.", ("Charged atoms", "Molecules", "Share"),
        _share_rows([*top, ("other", n - sum(c for _, c in top))], n),
        (f"{fmt_n(n)} of {fmt_n(len(ds))} molecules ({fmt_pct(n / len(ds))}) carry a formal charge; {fmt_pct(net / n)} "
         "of them have a non-zero net charge although QM9 contains only neutral molecules.",))
    rebuild_table = Table(
        "charge_rebuild", "Charged molecules after re-deriving bonds from the 3D geometry (total charge 0).",
        ("Outcome", "Molecules", "Share"),
        _share_rows([(label, outcomes[k]) for k, label in REBUILD_OUTCOMES], len(sample)),
        (f"Deterministic sample of every {every}th charged molecule ({fmt_n(len(sample))} molecules), "
         "RDKit rdDetermineBonds.",))
    return pattern_table, rebuild_table, examples


def class_balance_table(data):
    h = graph_histograms(*data.graphs(data.train_idx))
    rows = []
    for entity, key, labels in (("atom", "atom_types", ATOM_LIST), ("atom pair", "bond_types", BOND_LABELS)):
        rows += [(entity, *row) for row in _share_rows(zip(labels, h[key].tolist()), h[key].sum().item())]
    return Table("class_balance", "Class frequencies in the training split.", ("Entity", "Class", "Count", "Share"), rows,
                 ("Atom pairs include all pairs of real atoms, so 'none' is the majority class.",))


def atom_order_table(data):
    _, adj, mask = data.graphs(data.train_idx)
    i, j = torch.triu_indices(MAX_ATOMS, MAX_ATOMS, 1)
    bonded, consecutive, real = adj[:, i, j] > 0, (j - i) == 1, mask[:, i] & mask[:, j]
    share = lambda num, den: fmt_pct(num.sum().item() / den.sum().item())
    return Table("atom_order", "Bonds and atom order in the training split.", ("Quantity", "Value"),
                 [("bonds between consecutive atoms (i, i+1)", share(bonded & consecutive, bonded)),
                  ("P(bond | consecutive pair)", share(bonded & consecutive, consecutive & real)),
                  ("P(bond | other pair)", share(bonded & ~consecutive, ~consecutive & real))],
                 ("Atom order follows the canonical SMILES; only order-dependent models can exploit it.",))


def encoding_figure(data):
    k = next(int(i) for i in data.train_idx if int(data.masks[i].sum()) == 6 and 2 in data.edge_vecs[i].tolist()
             and 2 in data.atoms[i][data.masks[i]].tolist())
    atoms, adj, mask = (x[0] for x in data.graphs([k]))
    n = int(mask.sum())
    fig, (a, b, c) = figure(FULL_WIDTH, 2.3, 1, 3, gridspec_kw={"width_ratios": [1.1, 0.75, 1.2]})
    draw_molecule(a, atoms[:n].tolist(), adj[:n, :n].numpy(), _layout(atoms, adj, mask))
    a.axis("off")
    a.set_title("molecule")
    X = F.one_hot(atoms, NUM_ATOM_TYPES).numpy() * mask[:, None].numpy()
    for ax, matrix, xlabels, title in ((b, X, ATOM_LIST, "atoms X"), (c, adj.numpy(), range(MAX_ATOMS), "bond classes E")):
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(matrix.max(), 1), aspect="equal")
        ax.set_xticks(range(len(xlabels)), [str(x) for x in xlabels])
        ax.set_yticks(range(MAX_ATOMS), [f"{r}{'' if mask[r] else ' pad'}" for r in range(MAX_ATOMS)], fontsize=6)
        ax.axhspan(n - 0.5, MAX_ATOMS - 0.5, color=MUTED, alpha=0.35)
        ax.set_title(title)
    for r, s in zip(*np.nonzero(adj.numpy())):
        c.text(s, r, str(int(adj[r, s])), ha="center", va="center", fontsize=6, color="white")
    fig.tight_layout()
    return fig


def forward_noising(data, index, outfile, steps=8):
    atom_marginal, bond_marginal, _ = bootstrap_stats(data)
    i = data.train_idx[index]
    ts = torch.linspace(0, DIFFUSION_STEPS, steps).round().long()
    X0 = F.one_hot(data.atoms[i].long(), NUM_ATOM_TYPES).float().expand(steps, -1, -1)
    E0 = F.one_hot(data.edge_vecs[i].long(), NUM_BOND_TYPES).float().expand(steps, -1, -1)
    mask = data.masks[i].expand(steps, -1)
    torch.manual_seed(RANDOM_SEED)
    X, E = noise_graph(X0, E0, mask, ts.clamp(min=1), DiscreteDiffusion(atom_marginal, DIFFUSION_STEPS),
                       DiscreteDiffusion(bond_marginal, DIFFUSION_STEPS))
    X[0], E[0] = X0[0], E0[0]
    show(X.argmax(-1), edge_vector_to_adj(E.argmax(-1), MAX_ATOMS), mask, outfile=outfile,
         legends=[f"t = {t}" for t in ts.tolist()])


def export_data_quality(data, out_dir):
    out = Path(out_dir)
    pattern_table, rebuild_table, examples = charge_analysis()
    write_tables([pattern_table, rebuild_table, class_balance_table(data), atom_order_table(data)], out)
    graphs = [pair[0] for pair, _, _ in examples] + [pair[1] for pair, _, _ in examples]
    atoms, adj, mask = (torch.stack(x) for x in zip(*graphs))
    show(atoms, adj, mask, outfile=out / "figures" / "charged_examples.pdf", ncols=len(examples),
         legends=[e[1] for e in examples] + [e[2] for e in examples])
    show(*data.graphs(data.train_idx[:16]), outfile=out / "figures" / "train_examples.pdf")
    forward_noising(data, 0, out / "figures" / "forward_noising.pdf")
    save(encoding_figure(data), out / "figures" / "encoding.pdf")
    return [pattern_table, rebuild_table]
