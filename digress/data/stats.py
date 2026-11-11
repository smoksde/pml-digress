from collections import Counter
from typing import NamedTuple

from digress.config import MAX_ATOMS, QM9_TARGETS
from digress.data.chem import ATOM_LIST
from digress.data.dataset import SKIP_REASONS
from digress.data.scan import BOND_NAMES

REASON_LABELS = {
    "ok": "usable (passes our filter)",
    "unparsable": "RDKit cannot parse SMILES",
    "too_many_atoms": "more than 9 atoms after parsing",
    "unsupported_element": "element outside C, N, O, F",
    "charged": "non-zero formal charge",
    "conversion_failed": "graph conversion fails",
}
FUNNEL_STEPS = (  # the filter's checks in the order dataset.molecule_to_record applies them
    ("all QM9 molecules", None),
    ("parsed by RDKit", "unparsable"),
    ("at most 9 atoms", "too_many_atoms"),
    ("only C, N, O, F", "unsupported_element"),
    ("no formal charge", "charged"),
    ("converted to a graph", "conversion_failed"),
)
UNITS = {"mu": "D", "alpha": "a0^3", "homo": "eV", "lumo": "eV", "gap": "eV", "r2": "a0^2", "zpve": "eV",
         "u0": "eV", "u": "eV", "h": "eV", "g": "eV", "cv": "cal/(mol K)"}
PRETTY_UNITS = {**UNITS, "alpha": "a₀³", "r2": "a₀²"}
FIGURE_TARGETS = ("mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "u0", "cv")  # u, h, g look like u0
ROBUST_Z = 5.0  # outlier if |x - median| / (1.4826 * MAD) exceeds this


def usable_values(scan, column):
    return [v for r, v in zip(scan.reasons, column) if r == "ok"]


def outcome_counts(scan):
    counts = Counter(scan.reasons)
    return Counter({key: counts[key] for key in ("ok", *SKIP_REASONS)})


def filter_funnel(scan):
    counts, remaining, rows = outcome_counts(scan), scan.total, []
    for label, reason in FUNNEL_STEPS:
        removed = counts[reason] if reason else 0
        remaining -= removed
        rows.append((label, remaining, removed))
    return rows


def atom_count_table(scan, max_atoms=MAX_ATOMS):
    parsed = Counter(n for n in scan.n_atoms if n > 0)
    usable = Counter(usable_values(scan, scan.n_atoms))
    above = sum(c for n, c in parsed.items() if n > max_atoms)
    return [(n, parsed[n], usable[n]) for n in range(1, max_atoms + 1)], above


def element_table(scan):
    symbols = usable_values(scan, scan.symbols)
    atoms = Counter(a for s in symbols for a in s)
    molecules = Counter(a for s in symbols for a in set(s))
    total = sum(atoms.values())
    return [(el, atoms[el], atoms[el] / total, molecules[el] / len(symbols)) for el in ATOM_LIST]


def bond_table(scan):
    counts = scan.bond_counts.sum(0).tolist()
    total = sum(counts)
    return [(n, c, c / total) for n, c in zip(BOND_NAMES, counts)], total / int(scan.usable.sum())


def ring_table(scan):
    counts = Counter(usable_values(scan, scan.n_rings))
    return [(k, v, v / sum(counts.values())) for k, v in sorted(counts.items())]


def property_values(scan, name):
    return scan.y[scan.usable][:, QM9_TARGETS[name]]


class TargetSummary(NamedTuple):
    name: str
    mean: float
    std: float
    minimum: float
    median: float
    maximum: float
    skew: float
    low_fence: float
    high_fence: float
    outliers: int
    share: float


def summarize_target(scan, name):
    x = property_values(scan, name)
    median = x.median()
    mad = (x - median).abs().median() * 1.4826
    lo, hi = (median - ROBUST_Z * mad).item(), (median + ROBUST_Z * mad).item()
    n_out = int(((x < lo) | (x > hi)).sum())
    skew = (((x - x.mean()) / x.std()) ** 3).mean().item()
    return TargetSummary(name, x.mean().item(), x.std().item(), x.min().item(), median.item(), x.max().item(),
                         skew, lo, hi, n_out, n_out / len(x))
