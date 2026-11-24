"""
changes: adding in support for new model, new_model_config() moved from checkpoint.py.
create_mode() gets sizes from info. 
"""
import copy

from digress.config import DIGRESS_PRESETS, M1_MODEL_PARAMS
from digress.features import FeatureConfig
from digress.models.digress_transformer import DiGressTransformer
from digress.models.equivariant_mlp import EquivariantMLP
from digress.models.graph_transformer import GraphTransformer
from digress.models.mlp import MLP

MODEL_REGISTRY = {
    "mlp": MLP,
    "equivariant_mlp": EquivariantMLP,
    "simple_graph_transformer": GraphTransformer,
    "digress": DiGressTransformer,
}


def new_model_config(model_type, dataset, diffusion_steps, y_dim=0, hidden_dim=None, n_layers=None, dropout=None,
                     features=None):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type {model_type!r}; supported types: {', '.join(sorted(MODEL_REGISTRY))}")
    features = features or FeatureConfig()
    if model_type == "digress":
        params = copy.deepcopy(DIGRESS_PRESETS[dataset])
        if n_layers is not None:
            params["n_layers"] = n_layers
        if hidden_dim is not None:
            params["hidden_dims"]["dx"] = params["hidden_dims"]["dim_ffX"] = hidden_dim
        if dropout is not None:
            params["dropout"] = dropout
        params.update(diffusion_steps=diffusion_steps, y_dim=y_dim, features=features.to_dict())
        return {"type": model_type, "params": params}

    if features.any:
        raise ValueError("extra features are only supported by the 'digress' model")
    if dropout is not None:
        raise ValueError("--dropout is only supported by the 'digress' model")
    params = {"time_dim": M1_MODEL_PARAMS["time_dim"], "y_dim": y_dim}
    if model_type == "mlp" or hidden_dim is not None:
        params["hidden_dim"] = hidden_dim or M1_MODEL_PARAMS["hidden_dim"]
    if n_layers is not None:
        if model_type != "simple_graph_transformer":
            raise ValueError(f"--n-layers is not supported by {model_type!r}")
        params["num_layers"] = n_layers
    return {"type": model_type, "params": params}


def create_model(config, info):
    if config["type"] not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type {config['type']!r}; supported types: {', '.join(sorted(MODEL_REGISTRY))}"
        )
    cls = MODEL_REGISTRY[config["type"]]
    extra = {"info": info} if getattr(cls, "needs_info", False) else {}
    return cls(info.max_atoms, info.num_edges, info.num_atom_types, info.num_bond_types, **config["params"], **extra)
