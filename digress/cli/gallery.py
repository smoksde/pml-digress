from digress.config import MAX_ATOMS, PLOTS_DIR
from digress.data.dataset import get_dataset
from digress.data.graphs import edge_vector_to_adj
from digress.report.molecules import graphs_from_sample, show

if __name__ == "__main__":
    data = get_dataset()
    idx = data.train_idx[:16]
    adj = edge_vector_to_adj(data.edge_vecs[idx], MAX_ATOMS)
    show(graphs_from_sample(data.atoms[idx], adj, data.masks[idx]), outfile=PLOTS_DIR / "gallery.png")
