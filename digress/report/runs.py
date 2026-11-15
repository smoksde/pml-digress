import csv
import json
import math
from pathlib import Path

import torch

from digress.data.chem import ATOM_LIST
from digress.evaluation.metrics import failure_breakdown
from digress.report.molecules import show
from digress.report.style import FULL_WIDTH, HALF_WIDTH, INK_2, MUTED, SERIES, figure
from digress.report.tables import Table


def load_run(path, label=None):
    path = Path(path)

    def rows(name):
        if not (path / name).exists():
            return []
        with open(path / name) as f:
            return [{k: float(v) for k, v in r.items() if v != ""} for r in csv.DictReader(f)]

    test = path / "test_results.json"
    return {"dir": path, "name": label or path.name, "config": json.loads((path / "config.json").read_text()),
            "metrics": rows("metrics.csv"), "samples": rows("samples.csv"),
            "test": json.loads(test.read_text()) if test.exists() else None}


def _col(rows, key):
    return [r["step"] for r in rows if key in r], [r[key] for r in rows if key in r]


def _colors(runs):
    if len(runs) > len(SERIES):
        raise ValueError(f"compare at most {len(SERIES)} runs per report so colours stay distinguishable")
    return SERIES[: len(runs)]


def loss_curves(runs):
    fig, axes = figure(FULL_WIDTH, 2.0, 1, 3, sharex=True)
    for run, color in zip(runs, _colors(runs)):
        for ax, part in zip(axes, ("loss", "atom", "bond")):
            ax.plot(*_col(run["metrics"], f"train_{part}"), color=color, lw=1.0, alpha=0.45)
            ax.plot(*_col(run["metrics"], f"train_eval_{part}"), color=color, ls="--")
            ax.plot(*_col(run["metrics"], f"val_{part}"), color=color, label=run["name"])
    for ax, title in zip(axes, ("Total loss", "Atom loss", "Bond loss")):
        ax.set(title=title, xlabel="step")
        ax.grid(True, axis="y")
    axes[0].legend(title="solid: val\ndashed: train (fixed noise)\nfaint: train (running)")
    fig.tight_layout()
    return fig


SAMPLE_PANELS = (("valid_connected", "Valid and connected"), ("validity", "Valid (RDKit)"),
                 ("connected", "Connected"), ("novelty", "Novelty"), ("vun", "Valid, unique, novel"),
                 ("bonds_per_mol", "Bonds per molecule"), ("tv_atom_types", "Atom types: TV"),
                 ("tv_bond_types", "Bond types: TV"), ("tv_valence", "Valences: TV"))
SHARES = {"valid_connected", "validity", "connected", "novelty", "vun"}


def sample_quality(runs):
    if not any(r["samples"] for r in runs):
        return None
    fig, axes = figure(FULL_WIDTH, 4.6, 3, 3, sharex=True)
    for run, color in zip(runs, _colors(runs)):
        for ax, (key, _) in zip(axes.flat, SAMPLE_PANELS):
            ax.plot(*_col(run["samples"], key), color=color, marker="o", label=run["name"])
            if key == "bonds_per_mol" and run["config"].get("train_bonds_per_mol"):
                ax.axhline(run["config"]["train_bonds_per_mol"], color=INK_2, lw=0.8, ls="--")
    for i, (ax, (key, title)) in enumerate(zip(axes.flat, SAMPLE_PANELS)):
        ax.set_title(title)
        ax.grid(True, axis="y")
        if key in SHARES:
            ax.set_ylim(0, 1.05)
        if i >= 6:
            ax.set_xlabel("step")
    if len(runs) > 1:
        axes.flat[0].legend()
    fig.tight_layout()
    return fig


def _test_samples(run):
    path = run["dir"] / "test_samples.pt"
    if "test_samples" not in run and path.exists():
        run["test_samples"] = torch.load(path)
        run["breakdown"] = failure_breakdown(**run["test_samples"])
    return run.get("test_samples")


def _tested(runs, key):
    return [r for r in runs if r["test"] and key in r["test"]]


def test_loss_by_t(runs):
    tested = _tested(runs, "per_entry_reference")
    if not tested:
        return None
    ref = tested[0]["test"]
    ts = [int(k[1:]) for k in ref["per_entry_reference"]]
    fig, ax = figure(HALF_WIDTH * 1.2, 2.2)
    for run, color in zip(tested, _colors(tested)):
        ax.plot(ts, [run["test"][f"test_t{t}"] for t in ts], color=color, marker="o", label=run["name"])
    ax.plot(ts, list(ref["per_entry_reference"].values()), color=INK_2, lw=1.0, ls="--", label="per-entry only")
    ax.axhline(ref["marginal_floor"], color=MUTED, lw=1, ls=":", label="predict marginals")
    ax.set(xlabel="diffusion step t (higher = noisier)", ylabel="test loss")
    ax.grid(True, axis="y")
    ax.legend()
    return fig


RECALL_PANELS = (("bonds", ("single", "double", "triple")), ("atoms", ("N", "O", "F")))


def denoising_recall(runs):
    tested = _tested(runs, "denoising")
    if not tested:
        return None
    fig, axes = figure(FULL_WIDTH, 2.1, 1, 2, sharey=True)
    for ax, (part, labels) in zip(axes, RECALL_PANELS):
        for run, ls in zip(tested, ("-", "--", ":")):
            bins = run["test"]["denoising"]
            mids = [sum(int(x) for x in k[1:].split("-")) / 2 for k in bins]
            for i, (label, color) in enumerate(zip(labels, SERIES), start=1):
                ax.plot(mids, [b[part]["recall"][i] for b in bins.values()], color=color, ls=ls, marker="o", label=label if ls == "-" else None)
        ax.set(xlabel="diffusion step t", title=f"{part.capitalize()}: recall per class")
        ax.grid(True, axis="y")
        ax.legend()
    axes[0].set(ylabel="recall", ylim=(0, 1.05))
    if len(tested) > 1:
        fig.suptitle("  ".join(f"{s}: {r['name']}" for s, r in zip(("solid", "dashed", "dotted"), tested)),
                     fontsize=6.5, color=INK_2)
    fig.tight_layout()
    return fig


FIGURES = (loss_curves, sample_quality, test_loss_by_t, denoising_recall)


def failure_table(runs):
    rows = [(run["name"], *("-" if math.isnan(v) else f"{100 * v:.1f}" for v in run["breakdown"]["violation_rate"].values()))
            for run in runs if _test_samples(run) is not None]
    if not rows:
        return None
    return Table("valence_failures", "Generated atoms whose bond-order sum exceeds the neutral valence, by element (%).",
                 ("Run", *ATOM_LIST), rows, ("From the samples drawn by digress.cli.evaluate_test.",))


def failure_examples(run, outfile, k=16):
    samples = _test_samples(run)
    if samples is None:
        return False
    idx = [i for i, ok in enumerate(run["breakdown"]["valid"]) if not ok][:k]
    if idx:
        show(samples["atoms"][idx], samples["adj"][idx], samples["mask"][idx], outfile=outfile)
    return bool(idx)


def load_baselines(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


COMPARISON = (("validity", "Valid"), ("novelty", "Novel"), ("vun", "VUN"), ("tv_atom_types", "TV atoms"),
              ("tv_bond_types", "TV bonds"), ("tv_valence", "TV valence"))


def comparison_table(runs, baselines=None):
    rows = [(run["name"], f"{run['test']['test_loss']:.3f}", *(f"{run['test']['samples'][k]:.3f}" for k, _ in COMPARISON))
            for run in runs if run["test"]]
    if baselines and baselines["reference"] == "test":
        rows += [(b["name"], "-", *(f"{b[k]:.3f}" for k, _ in COMPARISON)) for b in baselines["baselines"].values()]
    if not rows:
        return None
    return Table("test_comparison", "Sample quality on the test split, 1,000 samples per generator.",
                 ("Generator", "Test loss", *(label for _, label in COMPARISON)), rows,
                 ("Models use the checkpoint with the lowest validation loss. VUN: valid, connected, unique and novel; "
                  "TV: total variation distance to the test molecules (0 = identical).",))
