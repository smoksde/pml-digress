import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from rdkit import Chem
from rdkit.Chem import rdDepictor

from digress.data.chem import ATOM_LIST, MAX_VALENCE, build_mol, graph_to_mol
from digress.report.style import BLUE, INK, INK_2, ORANGE, figure

ATOM_COLORS = {"C": INK, "N": BLUE, "O": "#e34948", "F": "#008300"}
BOND_OFFSETS = {1: (0.0,), 2: (-0.1, 0.1), 3: (-0.2, 0.0, 0.2)}


def stress_layout(n, edges, iters=300):
    D = np.full((n, n), np.inf)
    np.fill_diagonal(D, 0.0)
    for i, j in edges:
        D[i, j] = D[j, i] = 1.0
    for k in range(n):
        D = np.minimum(D, D[:, [k]] + D[[k], :])
    finite = np.isfinite(D)
    D[~finite] = D[finite].max() + 1.5
    J = np.eye(n) - 1.0 / n
    w, v = np.linalg.eigh(-0.5 * J @ (D**2) @ J)
    X = v[:, -2:] * np.sqrt(np.maximum(w[-2:], 1e-6)) + 1e-3 * np.random.default_rng(0).standard_normal((n, 2))
    W = np.where(D > 0, 1.0 / np.maximum(D, 1e-9) ** 2, 0.0)
    L_pinv = np.linalg.pinv(np.diag(W.sum(1)) - W)
    for _ in range(iters):
        dist = np.linalg.norm(X[:, None] - X[None], axis=-1)
        B = -np.where(dist > 1e-9, W * D / np.maximum(dist, 1e-9), 0.0)
        np.fill_diagonal(B, -B.sum(1))
        X = L_pinv @ B @ X
    X -= X.mean(0)
    return X @ np.linalg.svd(X, full_matrices=False)[2].T


def _layout(atoms, adj, mask):
    n = int(mask.sum())
    if n == 1:
        return np.zeros((1, 2))
    try:
        mol = build_mol(atoms, adj, mask)
        mol.UpdatePropertyCache(strict=False)
        rdDepictor.Compute2DCoords(mol)
        return mol.GetConformer().GetPositions()[:, :2] / 1.5
    except Exception:
        return stress_layout(n, [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j] > 0])


def draw_molecule(ax, atom_types, adj, pos, panel_size=1.6):
    pos = pos - pos.mean(0)
    r = max(np.abs(pos).max() + 0.8, 2.2)
    n = len(atom_types)
    for k in range(n):
        if adj[k].sum() > MAX_VALENCE[atom_types[k]]:
            ax.add_patch(Circle(pos[k], 0.38, fc="none", ec=ORANGE, lw=1.6, zorder=4))
    for i in range(n):
        for j in range(i + 1, n):
            d = pos[j] - pos[i]
            if adj[i, j] == 0 or np.linalg.norm(d) < 1e-9:
                continue
            normal = np.array([-d[1], d[0]]) / np.linalg.norm(d)
            for o in BOND_OFFSETS[int(adj[i, j])]:
                a, b = pos[i] + o * normal, pos[j] + o * normal
                ax.plot([a[0], b[0]], [a[1], b[1]], color=INK, lw=1.4, solid_capstyle="round", zorder=1)
    for k, a in enumerate(atom_types):
        s = ATOM_LIST[a]
        if s != "C" or not adj[k].any():
            ax.text(*pos[k], s, ha="center", va="center", fontsize=min(11, 14 * panel_size / r), fontweight="bold",
                    color=ATOM_COLORS[s],
                    zorder=3, bbox=dict(boxstyle="circle,pad=0.12", fc="white", ec="none"))
    ax.set(xlim=(-r, r), ylim=(-r, r), aspect="equal")


def _title(mol):
    if mol is None:
        return "invalid"
    smiles = Chem.MolToSmiles(mol)
    return smiles if len(Chem.GetMolFrags(mol)) == 1 else smiles + " (disconn.)"


def show(atoms, adj, mask, outfile=None, ncols=8, legends=None, panel_size=1.6):
    atoms, adj, mask = atoms.cpu(), adj.cpu(), mask.cpu().bool()
    if len(atoms) == 0:
        raise ValueError("show() got no molecules")
    ncols = min(ncols, len(atoms))
    nrows = math.ceil(len(atoms) / ncols)
    fig, axes = figure(panel_size * ncols, panel_size * nrows + 0.3, nrows, ncols, squeeze=False)
    n_valid = 0
    for k, ax in enumerate(axes.flat):
        ax.axis("off")
        if k >= len(atoms):
            continue
        n = int(mask[k].sum())
        mol = graph_to_mol(atoms[k], adj[k], mask[k])
        n_valid += mol is not None
        draw_molecule(ax, atoms[k, :n].tolist(), adj[k, :n, :n].numpy(), _layout(atoms[k], adj[k], mask[k]),
                      panel_size)
        title = legends[k] if legends else _title(mol)
        ax.set_title(title if len(title) <= 22 else title[:21] + "…", fontsize=7, fontweight="normal", loc="center",
                     color=INK if legends or mol is not None else ORANGE)
    handles = [Line2D([], [], marker=f"${s}$", color=ATOM_COLORS[s], ls="", ms=7, label=s) for s in ATOM_LIST]
    handles.append(Line2D([], [], marker="o", mfc="none", color=ORANGE, ls="", ms=8, label="valence violated"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=7, labelcolor=INK_2,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.3 / (panel_size * nrows + 0.3), 1, 1))
    if outfile:
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=200)
    plt.close(fig)
    print(f"{n_valid}/{len(atoms)} valid")
    return fig
