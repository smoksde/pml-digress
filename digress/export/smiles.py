from collections import Counter

from rdkit import Chem

from digress.data.chem import build_mol, graph_to_mol


def molecule_smiles(atom_idx, bond_adj, mask):
    mol = graph_to_mol(atom_idx, bond_adj, mask)
    if mol is not None:
        return Chem.MolToSmiles(mol), "valid" if len(Chem.GetMolFrags(mol)) == 1 else "valid_disconnected"
    try:
        raw = build_mol(atom_idx, bond_adj, mask)
        raw.UpdatePropertyCache(strict=False)
        return Chem.MolToSmiles(raw), "invalid"
    except Exception:
        return "", "invalid"


def write_smi(atoms, adj, mask, path):
    counts = Counter()
    with open(path, "w") as f:
        for a, b, m in zip(atoms, adj, mask):
            smiles, status = molecule_smiles(a, b, m)
            counts[status] += 1
            f.write(f"{smiles}\t{status}\n")
    return counts
