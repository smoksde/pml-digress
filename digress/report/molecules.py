from pathlib import Path

import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Draw

from digress.data.chem import NUM_ATOM_TYPES, NUM_BOND_TYPES, graph_to_mol, mol_for_drawing
from digress.data.graphs import adj_to_edge_vector, edge_vector_to_adj

INVALID_COLOR = (1.0, 0.75, 0.75)


def graphs_from_sample(atoms, bonds, mask):
    edge_vecs = adj_to_edge_vector(bonds)
    return [(F.one_hot(a, NUM_ATOM_TYPES).float(), F.one_hot(e, NUM_BOND_TYPES).float(), m)
            for a, e, m in zip(atoms, edge_vecs, mask)]


def _drawable(mol):
    try:
        Draw.MolToImage(mol, size=(60, 60))
        return True
    except RuntimeError:
        return False


def show(graphs, outfile=None, mols_per_row=4, sub_img_size=(220, 220), legends=None):
    if len(graphs) == 0:
        raise ValueError("show() got an empty list")
    mols, caps, atom_hl, bond_hl = [], [], [], []
    n_valid = n_undrawable = 0
    for k, (X, E, mask) in enumerate(graphs):
        atom_idx, bond_adj = X.argmax(-1), edge_vector_to_adj(E.argmax(-1).unsqueeze(0), X.shape[0])[0]
        mol = graph_to_mol(atom_idx, bond_adj, mask)
        valid = mol is not None
        if not valid:
            mol = mol_for_drawing(atom_idx, bond_adj, mask)
            if not _drawable(mol):
                n_undrawable += 1
                continue
        n_valid += valid
        mols.append(mol)
        caps.append(legends[k] if legends else Chem.MolToSmiles(mol) if valid else "invalid")
        atom_hl.append([] if valid else list(range(mol.GetNumAtoms())))  # invalid molecules drawn in red
        bond_hl.append([] if valid else list(range(mol.GetNumBonds())))

    if not mols:
        print(f"nothing to draw ({n_undrawable} graphs could not be laid out)")
        return None
    img = Draw.MolsToGridImage(
        mols, molsPerRow=mols_per_row, subImgSize=sub_img_size, legends=caps, highlightAtomLists=atom_hl,
        highlightBondLists=bond_hl, highlightAtomColors=[dict.fromkeys(a, INVALID_COLOR) for a in atom_hl],
        highlightBondColors=[dict.fromkeys(b, INVALID_COLOR) for b in bond_hl],
    )
    if outfile:
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        img.save(outfile)
    print(f"{n_valid}/{len(graphs)} valid" + (f", {n_undrawable} not drawable (skipped)" if n_undrawable else ""))
    return img
