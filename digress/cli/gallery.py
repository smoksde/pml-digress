import argparse

import torch
import torch.nn.functional as F

from digress.config import DIFFUSION_STEPS, MAX_ATOMS, PLOTS_DIR, RANDOM_SEED
from digress.data.chem import NUM_ATOM_TYPES, NUM_BOND_TYPES
from digress.data.dataset import get_dataset
from digress.data.graphs import edge_vector_to_adj
from digress.data.marginals import bootstrap_stats
from digress.diffusion.discrete import DiscreteDiffusion, noise_graph
from digress.report.molecules import graphs_from_sample, show


def forward_noising(data, index, steps=8):
    atom_marginal, bond_marginal, _ = bootstrap_stats(NUM_ATOM_TYPES, NUM_BOND_TYPES)
    i = data.train_idx[index]
    ts = torch.linspace(0, DIFFUSION_STEPS, steps).round().long()
    X0 = F.one_hot(data.atoms[i], NUM_ATOM_TYPES).float().expand(steps, -1, -1)
    E0 = F.one_hot(data.edge_vecs[i], NUM_BOND_TYPES).float().expand(steps, -1, -1)
    mask = data.masks[i].expand(steps, -1)
    torch.manual_seed(RANDOM_SEED)
    X, E = noise_graph(X0, E0, mask, ts.clamp(min=1), DiscreteDiffusion(atom_marginal, DIFFUSION_STEPS),
                       DiscreteDiffusion(bond_marginal, DIFFUSION_STEPS))
    X[0], E[0] = X0[0], E0[0]
    graphs = [(X[k], E[k], mask[k]) for k in range(steps)]
    show(graphs, outfile=PLOTS_DIR / "forward_noising.png", mols_per_row=steps, sub_img_size=(180, 180),
         legends=[f"t = {t}" for t in ts.tolist()])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--noising", action="store_true", help="draw one train molecule at increasing noise levels")
    parser.add_argument("--index", type=int, default=0, help="which train molecule to noise")
    args = parser.parse_args()
    data = get_dataset()
    if args.noising:
        forward_noising(data, args.index)
    else:
        idx = data.train_idx[:16]
        adj = edge_vector_to_adj(data.edge_vecs[idx], MAX_ATOMS)
        show(graphs_from_sample(data.atoms[idx], adj, data.masks[idx]), outfile=PLOTS_DIR / "gallery.png")
