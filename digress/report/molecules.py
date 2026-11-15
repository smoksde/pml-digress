import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from rdkit import Chem
from rdkit.Chem import rdDepictor

from digress.data.chem import ATOM_LIST, MAX_VALENCE, build_mol, graph_to_mol
from digress.report.style import BLUE, INK, INK_2, ORANGE, figure, save

ATOM_COLORS = {"C": INK, "N": BLUE, "O": "#e34948", "F": "#008300"}
BOND_OFFSETS = {1: (0.0,), 2: (-0.1, 0.1), 3: (-0.2, 0.0, 0.2)}


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
        return None


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
            ax.text(*pos[k], s, ha="center", va="center", fontsize=min(9, 12 * panel_size / r), fontweight="bold",
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
        pos = _layout(atoms[k], adj[k], mask[k])
        if pos is None:
            ax.set_title("no layout", fontsize=6, fontweight="normal", loc="center", color=ORANGE)
            continue
        draw_molecule(ax, atoms[k, :n].tolist(), adj[k, :n, :n].numpy(), pos,
                      panel_size)
        title = legends[k] if legends else _title(mol)
        ax.set_title(title if len(title) <= 22 else title[:21] + "…", fontsize=6, fontweight="normal", loc="center",
                     color=INK if legends or mol is not None else ORANGE)
    handles = [Line2D([], [], marker=f"${s}$", color=ATOM_COLORS[s], ls="", ms=7, label=s) for s in ATOM_LIST]
    handles.append(Line2D([], [], marker="o", mfc="none", color=ORANGE, ls="", ms=8, label="valence violated"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=6, labelcolor=INK_2,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.3 / (panel_size * nrows + 0.3), 1, 1))
    if outfile:
        save(fig, Path(outfile), dpi=200)
    plt.close(fig)
    print(f"{n_valid}/{len(atoms)} valid")
    return fig
