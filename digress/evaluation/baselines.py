import random

import torch

from digress.config import HEURISTIC_P_RING, HEURISTIC_P_UPGRADE, MAX_ATOMS, RANDOM_SEED
from digress.data.chem import MAX_VALENCE
from digress.data.graphs import edge_vector_to_adj
from digress.diffusion.sampling import _init_xt

MAX_REDRAWS = 1000


def sample_prior(marginals, num_samples, seed=RANDOM_SEED):
    torch.manual_seed(seed)
    atoms, bonds, mask, _ = _init_xt(marginals, num_samples)
    return atoms.cpu(), edge_vector_to_adj(bonds, MAX_ATOMS).cpu(), mask.cpu()


def _spanning_tree(atom_types, rng):
    free = [MAX_VALENCE[a] for a in atom_types]
    order = sorted(range(len(atom_types)), key=lambda k: -free[k])
    placed, bonds = [order[0]], {}
    for k in order[1:]:
        candidates = [p for p in placed if free[p] > 0]
        if not candidates:
            return None
        p = rng.choice(candidates)
        bonds[tuple(sorted((p, k)))] = 1
        free[p] -= 1
        free[k] -= 1
        placed.append(k)
    return bonds, free


def _heuristic_molecule(n, atom_marginal, rng, gen, p_upgrade, p_ring):
    for _ in range(MAX_REDRAWS):
        atom_types = torch.multinomial(
            atom_marginal, n, replacement=True, generator=gen
        ).tolist()
        tree = _spanning_tree(atom_types, rng)
        if tree is not None:
            break
    else:
        raise RuntimeError(f"no spanning tree for {n} atoms after {MAX_REDRAWS} draws")
    bonds, free = tree
    for i, j in list(bonds):
        while (
            free[i] > 0
            and free[j] > 0
            and bonds[(i, j)] < 3
            and rng.random() < p_upgrade
        ):
            bonds[(i, j)] += 1
            free[i] -= 1
            free[j] -= 1
    for i in range(n):
        for j in range(i + 1, n):
            if (
                (i, j) not in bonds
                and free[i] > 0
                and free[j] > 0
                and rng.random() < p_ring / n
            ):
                bonds[(i, j)] = 1
                free[i] -= 1
                free[j] -= 1
    return atom_types, bonds


def sample_valence_heuristic(
    marginals,
    num_samples,
    seed=RANDOM_SEED,
    p_upgrade=HEURISTIC_P_UPGRADE,
    p_ring=HEURISTIC_P_RING,
):
    rng, gen = random.Random(seed), torch.Generator().manual_seed(seed)
    atom_marginal = marginals["atom_marginal"].cpu()
    counts = torch.multinomial(
        marginals["atom_count_dist"].cpu(), num_samples, replacement=True, generator=gen
    )
    atoms = torch.zeros(num_samples, MAX_ATOMS, dtype=torch.long)
    adj = torch.zeros(num_samples, MAX_ATOMS, MAX_ATOMS, dtype=torch.long)
    mask = torch.zeros(num_samples, MAX_ATOMS, dtype=torch.bool)
    for s, n in enumerate(counts.tolist()):
        atom_types, bonds = _heuristic_molecule(
            n, atom_marginal, rng, gen, p_upgrade, p_ring
        )
        atoms[s, :n] = torch.tensor(atom_types)
        mask[s, :n] = True
        for (i, j), order in bonds.items():
            adj[s, i, j] = adj[s, j, i] = order
    return atoms, adj, mask
