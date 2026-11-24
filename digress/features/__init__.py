"""
new folder/ file: features/__init__.py
Extra input features computed from the noisy graph at every denoising step.
FeatureConfig stored by checkpoint so for a run we have what was used.
ExtraFeatures.dims sets the # of input channels for model, ExtraFeatures.__call__ returns the features as tensors.

--> right now: "none" -> no extra features,

node_index -> one-hot of node's position in (padded) tensor, breaking permutation equivariance --> test out the differences?

todo: add structural features: structural "cycles", "eigenvalues", "all" and molecular (per-atom formal charge estimate, current valence, molecular weight) <- see author code

"""
from dataclasses import asdict, dataclass

import torch

STRUCTURAL = ("none", "cycles", "eigenvalues", "all")
#todo:: (dx, de, dy) each feature group adds when implemented
PLANNED_DIMS = {    # for later when integrating the extra features
    "cycles": (3, 0, 5),
    "eigenvalues": (3, 0, 11),
    "all": (6, 0, 11),
    "molecular": (2, 0, 1),
}


@dataclass(frozen=True)
class FeatureConfig:
    structural: str = "none"
    molecular: bool = False
    node_index: bool = False

    def __post_init__(self):
        if self.structural not in STRUCTURAL:
            raise ValueError(f"unknown structural features {self.structural!r}; choose from {STRUCTURAL}")

    @classmethod
    def from_dict(cls, d):
        return cls(**(d or {}))

    def to_dict(self):
        return asdict(self)

    @property
    def any(self):
        return self.structural != "none" or self.molecular or self.node_index

# Extra Features like structural cycles, connected components, ...
class ExtraFeatures:
    def __init__(self, config, info):
        self.config, self.info = config, info
        if config.structural != "none":
            raise NotImplementedError(
                f"structural features {config.structural!r} planned but not implemented yet (TODO)"
                f"(will add {PLANNED_DIMS[config.structural]} (dx, de, dy) channels); use --extra-features none")
        if config.molecular:
            raise NotImplementedError(
                f"molecular features planned but not implemented yet (will add {PLANNED_DIMS['molecular']} "
                "(dx, de, dy) channels); use without --molecular-features")

    @property
    def dims(self):
        dx = self.info.max_atoms if self.config.node_index else 0
        return dx, 0, 0

    def __call__(self, X, E, node_mask):
        """
        X: (bs, n, dx) one-hot noisy atoms, E: (bs, n, n, de) one-hot noisy bonds (zero rows = padding / diagonal).
        node_mask: (bs, n) (bool). Returns extra (bs, n, dx'), (bs, n, n, de'), (bs, dy') tensors
        """
        bs, n = node_mask.shape
        xs = []
        if self.config.node_index:
            eye = torch.eye(n, device=X.device, dtype=X.dtype)
            xs.append(eye.expand(bs, n, n) * node_mask.unsqueeze(-1))
        x_extra = torch.cat(xs, -1) if xs else X.new_zeros(bs, n, 0)
        return x_extra, E.new_zeros(bs, n, n, 0), X.new_zeros(bs, 0)
