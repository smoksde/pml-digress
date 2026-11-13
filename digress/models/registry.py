from digress.config import MAX_ATOMS, NUM_EDGES
from digress.data.chem import NUM_ATOM_TYPES, NUM_BOND_TYPES
from digress.models.mlp import MLP
from digress.models.equivariant_mlp import EquivariantMLP
from digress.models.graph_transformer import GraphTransformer

MODEL_REGISTRY = {"mlp": MLP, "equivariant_mlp": EquivariantMLP, "simple_graph_transformer": GraphTransformer}


def create_model(config):
    if config["type"] not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type {config['type']!r}; supported types: {', '.join(sorted(MODEL_REGISTRY))}"
        )
    return MODEL_REGISTRY[config["type"]](
        MAX_ATOMS, NUM_EDGES, NUM_ATOM_TYPES, NUM_BOND_TYPES, **config["params"]
    )
