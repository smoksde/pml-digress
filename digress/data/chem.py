import torch
from rdkit import Chem

ATOM_LIST = ["C", "N", "O", "F"]
MAX_VALENCE = [4, 3, 2, 1]
NUM_ATOM_TYPES = len(ATOM_LIST)

BOND_ORDER = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE]
NUM_BOND_TYPES = len(BOND_ORDER) + 1


def mol_to_graph(mol, max_atoms):
    mol = Chem.Mol(mol)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    atom_idx = torch.zeros(max_atoms, dtype=torch.long)
    mask = torch.zeros(max_atoms, dtype=torch.bool)
    for i, atom in enumerate(mol.GetAtoms()):
        atom_idx[i], mask[i] = ATOM_LIST.index(atom.GetSymbol()), True
    bond_adj = torch.zeros((max_atoms, max_atoms), dtype=torch.long)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        cls = BOND_ORDER.index(bond.GetBondType()) + 1
        bond_adj[i, j] = bond_adj[j, i] = cls
    return atom_idx, mask, bond_adj


def build_mol(atom_idx, bond_adj, mask):
    mol = Chem.RWMol()
    idx_map = {}
    for i, a in enumerate(atom_idx.tolist()):
        if mask[i]:
            idx_map[i] = mol.AddAtom(Chem.Atom(ATOM_LIST[a]))
    n = atom_idx.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            b = bond_adj[i, j].item()
            if b != 0 and i in idx_map and j in idx_map:
                mol.AddBond(idx_map[i], idx_map[j], BOND_ORDER[b - 1])
    return mol.GetMol()


def graph_to_mol(atom_idx, bond_adj, mask):
    mol = build_mol(atom_idx, bond_adj, mask)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol


def flat_smiles(mol):
    return Chem.MolToSmiles(mol, isomericSmiles=False)
