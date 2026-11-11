import csv
import json
from pathlib import Path

from digress.report.tables import Table
from digress.report.style import FULL_WIDTH, HALF_WIDTH, INK_2, MUTED, SERIES, figure


def load_run(path):
    path = Path(path)

    def rows(name):
        if not (path / name).exists():
            return []
        with open(path / name) as f:
            return [{k: float(v) for k, v in r.items() if v != ""} for r in csv.DictReader(f)]

    return {"name": path.name, "config": json.loads((path / "config.json").read_text()),
            "metrics": rows("metrics.csv"), "samples": rows("samples.csv")}


def _col(rows, key):
    return [r["step"] for r in rows if key in r], [r[key] for r in rows if key in r]


def _colors(runs):
    if len(runs) > len(SERIES):
        raise ValueError(f"compare at most {len(SERIES)} runs per report so colours stay distinguishable")
    return SERIES[: len(runs)]


def loss_curves(runs):
    fig, axes = figure(FULL_WIDTH, 2.3, 1, 3, sharex=True)
    for run, color in zip(runs, _colors(runs)):
        for ax, part in zip(axes, ("loss", "atom", "bond")):
            ax.plot(*_col(run["metrics"], f"train_{part}"), color=color, lw=1.2, alpha=0.45)
            ax.plot(*_col(run["metrics"], f"val_{part}"), color=color, lw=1.6, label=run["name"])
    for ax, title in zip(axes, ("Total loss", "Atom loss", "Bond loss")):
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(True, axis="y")
    axes[0].legend(title="solid: val, faint: train", title_fontsize=7)
    fig.tight_layout()
    return fig


def loss_by_t(runs):
    fig, axes = figure(FULL_WIDTH if len(runs) > 1 else HALF_WIDTH * 1.3, 2.4, 1, len(runs), squeeze=False)
    for ax, run in zip(axes[0], runs):
        rows = run["metrics"]
        keys = sorted((k for k in rows[0] if k.startswith("val_t")), key=lambda k: int(k[5:]))
        ts = [int(k[5:]) for k in keys]
        picks = sorted({0, len(rows) // 4, len(rows) // 2, len(rows) - 1})
        shades = [MUTED, "#8fb4e6", "#4f8fdc", SERIES[0]][-len(picks):]
        for i, shade in zip(picks, shades):
            ax.plot(ts, [rows[i][k] for k in keys], color=shade, lw=1.6, marker="o", ms=3,
                    label=f"step {int(rows[i]['step']):,}")
        ax.set_xlabel("diffusion step t (higher = noisier)")
        ax.set_ylabel("val loss")
        ax.set_title(f"Loss by noise level: {run['name']}")
        ax.grid(True, axis="y")
        ax.legend()
    fig.tight_layout()
    return fig


SAMPLE_PANELS = (("valid_connected", "Valid and connected"), ("validity", "Valid (RDKit)"),
                 ("connected", "Connected"), ("uniqueness", "Uniqueness"), ("novelty", "Novelty"),
                 ("fail_valence", "Valence failures"), ("bonds_per_mol", "Bonds per molecule"),
                 ("tv_atom_types", "Atom types: TV to train"), ("tv_bond_types", "Bond types: TV to train"))
SHARES = {"valid_connected", "validity", "connected", "uniqueness", "novelty", "fail_valence"}


def sample_quality(runs):
    if not any(r["samples"] for r in runs):
        return None
    fig, axes = figure(FULL_WIDTH, 5.4, 3, 3, sharex=True)
    for run, color in zip(runs, _colors(runs)):
        for ax, (key, _) in zip(axes.flat, SAMPLE_PANELS):
            ax.plot(*_col(run["samples"], key), color=color, lw=1.6, marker="o", ms=3, label=run["name"])
        ref = run["config"].get("train_bonds_per_mol")
        if ref:
            axes.flat[6].axhline(ref, color=INK_2, lw=0.8, ls="--")
            axes.flat[6].text(0.02, ref, " training", transform=axes.flat[6].get_yaxis_transform(),
                              va="bottom", fontsize=7, color=INK_2)
    for i, (ax, (key, title)) in enumerate(zip(axes.flat, SAMPLE_PANELS)):
        ax.set_title(title)
        ax.grid(True, axis="y")
        if key in SHARES:
            ax.set_ylim(0, 1.05)
        if i >= 6:
            ax.set_xlabel("step")
    if len(runs) > 1:
        axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    return fig


def diagnostics(runs):
    fig, axes = figure(FULL_WIDTH, 2.1, 1, 2)
    for run, color in zip(runs, _colors(runs)):
        axes[0].plot(*_col(run["metrics"], "grad_norm"), color=color, lw=1.4, label=run["name"])
        axes[1].plot(*_col(run["metrics"], "steps_per_sec"), color=color, lw=1.4, label=run["name"])
    axes[0].set_title("Gradient norm (interval mean)")
    axes[1].set_title("Throughput (steps/s)")
    for ax in axes:
        ax.set_xlabel("step")
        ax.grid(True, axis="y")
    if len(runs) > 1:
        axes[0].legend()
    fig.tight_layout()
    return fig


FIGURES = (loss_curves, loss_by_t, sample_quality, diagnostics)


def summary_table(runs):
    rows = []
    for run in runs:
        m, s, c = run["metrics"], run["samples"], run["config"]
        best = min(m, key=lambda r: r["val_loss"])
        last = s[-1] if s else {}
        rows.append((run["name"], f"{int(m[-1]['step']):,}", f"{c['lr']:g}", f"{c['batch_size']}",
                     f"{m[-1]['val_loss']:.4f}", f"{best['val_loss']:.4f} @ {int(best['step']):,}",
                     *(f"{last[k]:.2f}" if k in last else "-" for k in ("valid_connected", "validity", "uniqueness", "novelty")),
                     f"{m[-1]['elapsed_min']:.0f}"))
    return Table("training_runs", "Training runs: final and best validation loss and final sample quality.",
                 ("Run", "Steps", "LR", "Batch", "Final val", "Best val", "Valid+conn.", "Validity", "Uniqueness", "Novelty",
                  "Minutes"),
                 rows, ("Sample quality from the last evaluation. Valid+conn.: RDKit-valid and a single fragment; uniqueness "
                  "and novelty are computed on those, comparing stereo-free SMILES.",))
