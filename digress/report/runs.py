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
            return [
                {k: float(v) for k, v in r.items() if v != ""}
                for r in csv.DictReader(f)
            ]

    test = path / "test_results.json"
    return {
        "name": path.name,
        "config": json.loads((path / "config.json").read_text()),
        "metrics": rows("metrics.csv"),
        "samples": rows("samples.csv"),
        "test": json.loads(test.read_text()) if test.exists() else None,
    }


def _col(rows, key):
    return [r["step"] for r in rows if key in r], [r[key] for r in rows if key in r]


def _colors(runs):
    if len(runs) > len(SERIES):
        raise ValueError(
            f"compare at most {len(SERIES)} runs per report so colours stay distinguishable"
        )
    return SERIES[: len(runs)]


def loss_curves(runs):
    fig, axes = figure(FULL_WIDTH, 2.3, 1, 3, sharex=True)
    for run, color in zip(runs, _colors(runs)):
        for ax, part in zip(axes, ("loss", "atom", "bond")):
            ax.plot(
                *_col(run["metrics"], f"train_{part}"), color=color, lw=1.2, alpha=0.45
            )
            ax.plot(*_col(run["metrics"], f"train_eval_{part}"), color=color, lw=1.4, ls="--")
            ax.plot(
                *_col(run["metrics"], f"val_{part}"),
                color=color,
                lw=1.6,
                label=run["name"],
            )
    for ax, title in zip(axes, ("Total loss", "Atom loss", "Bond loss")):
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(True, axis="y")
    axes[0].legend(title="solid: val\ndashed: train (fixed noise)\nfaint: train (running)", title_fontsize=7)
    fig.tight_layout()
    return fig


def loss_by_t(runs):
    fig, axes = figure(
        FULL_WIDTH if len(runs) > 1 else HALF_WIDTH * 1.3,
        2.4,
        1,
        len(runs),
        squeeze=False,
    )
    for ax, run in zip(axes[0], runs):
        rows = run["metrics"]
        keys = sorted(
            (k for k in rows[0] if k.startswith("val_t")), key=lambda k: int(k[5:])
        )
        ts = [int(k[5:]) for k in keys]
        picks = sorted({0, len(rows) // 4, len(rows) // 2, len(rows) - 1})
        shades = [MUTED, "#8fb4e6", "#4f8fdc", SERIES[0]][-len(picks) :]
        for i, shade in zip(picks, shades):
            ax.plot(
                ts,
                [rows[i][k] for k in keys],
                color=shade,
                lw=1.6,
                marker="o",
                ms=3,
                label=f"step {int(rows[i]['step']):,}",
            )
        ax.set_xlabel("diffusion step t (higher = noisier)")
        ax.set_ylabel("val loss")
        ax.set_title(f"Loss by noise level: {run['name']}")
        ax.grid(True, axis="y")
        ax.legend()
    fig.tight_layout()
    return fig


SAMPLE_PANELS = (
    ("valid_connected", "Valid and connected"),
    ("validity", "Valid (RDKit)"),
    ("connected", "Connected"),
    ("uniqueness", "Uniqueness"),
    ("novelty", "Novelty"),
    ("vun", "Valid, unique, novel"),
    ("fail_valence", "Valence failures"),
    ("bonds_per_mol", "Bonds per molecule"),
    ("tv_n_atoms", "Atom counts: total variation"),
    ("tv_atom_types", "Atom types: total variation"),
    ("tv_bond_types", "Bond types: total variation"),
    ("tv_valence", "Valences: total variation"),
)
SHARES = {
    "valid_connected",
    "validity",
    "connected",
    "uniqueness",
    "novelty",
    "vun",
    "fail_valence",
}
BONDS_PANEL = [k for k, _ in SAMPLE_PANELS].index("bonds_per_mol")


def sample_quality(runs):
    if not any(r["samples"] for r in runs):
        return None
    fig, axes = figure(FULL_WIDTH, 7.2, 4, 3, sharex=True)
    for run, color in zip(runs, _colors(runs)):
        for ax, (key, _) in zip(axes.flat, SAMPLE_PANELS):
            ax.plot(
                *_col(run["samples"], key),
                color=color,
                lw=1.6,
                marker="o",
                ms=3,
                label=run["name"],
            )
        ref = run["config"].get("train_bonds_per_mol")
        if ref:
            axes.flat[BONDS_PANEL].axhline(ref, color=INK_2, lw=0.8, ls="--")
            axes.flat[BONDS_PANEL].text(
                0.02,
                ref,
                " training",
                transform=axes.flat[BONDS_PANEL].get_yaxis_transform(),
                va="bottom",
                fontsize=7,
                color=INK_2,
            )
    for i, (ax, (key, title)) in enumerate(zip(axes.flat, SAMPLE_PANELS)):
        ax.set_title(title)
        ax.grid(True, axis="y")
        if key in SHARES:
            ax.set_ylim(0, 1.05)
        if key.startswith("tv_"):
            ax.set_ylabel("distance to val", fontsize=7)
        if i >= len(SAMPLE_PANELS) - 3:
            ax.set_xlabel("step")
    if len(runs) > 1:
        axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    return fig


def diagnostics(runs):
    fig, axes = figure(FULL_WIDTH, 2.1, 1, 2)
    for run, color in zip(runs, _colors(runs)):
        axes[0].plot(
            *_col(run["metrics"], "grad_norm"), color=color, lw=1.4, label=run["name"]
        )
        axes[1].plot(
            *_col(run["metrics"], "steps_per_sec"),
            color=color,
            lw=1.4,
            label=run["name"],
        )
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


def load_baselines(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def _baseline_summary_rows(baselines):
    rows = []
    for b in baselines["baselines"].values():
        scores = [
            f"{b[k]:.2f}" for k in ("valid_connected", "uniqueness", "novelty", "vun")
        ]
        if baselines["reference"] == "test":
            rows.append(
                (
                    b["name"],
                    *["-"] * 9,
                    "-",
                    f"{b['valid_connected']:.2f}",
                    f"{b['vun']:.2f}",
                    "-",
                )
            )
        else:
            rows.append((b["name"], *["-"] * 5, *scores, "-", "-", "-", "-"))
    return rows


def summary_table(runs, baselines=None):
    rows = []
    for run in runs:
        m, s, c = run["metrics"], run["samples"], run["config"]
        best = min(m, key=lambda r: r["val_loss"])
        last = s[-1] if s else {}
        test = run["test"] or {}
        test_samples = test.get("samples", {})
        rows.append(
            (
                run["name"],
                f"{int(m[-1]['step']):,}",
                f"{c['lr']:g}",
                f"{c['batch_size']}",
                f"{m[-1]['val_loss']:.4f}",
                f"{best['val_loss']:.4f} @ {int(best['step']):,}",
                *(
                    f"{last[k]:.2f}" if k in last else "-"
                    for k in ("valid_connected", "uniqueness", "novelty", "vun")
                ),
                f"{test['test_loss']:.4f}" if test else "-",
                *(
                    f"{test_samples[k]:.2f}" if k in test_samples else "-"
                    for k in ("valid_connected", "vun")
                ),
                f"{m[-1]['elapsed_min']:.0f}",
            )
        )
    if baselines:
        rows += _baseline_summary_rows(baselines)
    return Table(
        "training_runs",
        "Training runs: validation loss, last sample quality, and the final test evaluation.",
        (
            "Run",
            "Steps",
            "LR",
            "Batch",
            "Final val",
            "Best val",
            "Valid+conn.",
            "Uniqueness",
            "Novelty",
            "VUN",
            "Test loss",
            "Test valid+conn.",
            "Test VUN",
            "Minutes",
        ),
        rows,
        (
            "Sample quality from the last evaluation during training. Valid+conn.: RDKit-valid and a single "
            "fragment; uniqueness and novelty are computed on those, comparing stereo-free SMILES. VUN: valid, "
            "connected, unique and novel, as a share of all samples. Total variation distance (TV, tv_* columns in "
            "samples.csv): 0.5 * sum |p - q| between generated and validation histograms; 0 = identical "
            "distributions, 1 = no overlap. Test columns: best.pt scored once on the "
            "test split by digress.cli.evaluate_test ('-' if not run yet). Prior (untrained) and Valence heuristic: "
            "non-learned baselines from digress.cli.evaluate_baselines.",
        ),
    )


COMPARISON = (
    ("valid_connected", "Valid+conn."),
    ("validity", "Validity"),
    ("uniqueness", "Uniqueness"),
    ("novelty", "Novelty"),
    ("vun", "VUN"),
    ("tv_n_atoms", "TV atom counts"),
    ("tv_atom_types", "TV atom types"),
    ("tv_bond_types", "TV bond types"),
    ("tv_valence", "TV valences"),
)


def comparison_table(runs, baselines=None):
    rows = [
        (run["name"], *(f"{run['test']['samples'][k]:.3f}" for k, _ in COMPARISON))
        for run in runs
        if run["test"]
    ]
    if baselines and baselines["reference"] == "test":
        rows += [
            (b["name"], *(f"{b[k]:.3f}" for k, _ in COMPARISON))
            for b in baselines["baselines"].values()
        ]
    if not rows:
        return None
    return Table(
        "test_comparison",
        "Sample quality on the test split: trained models versus non-learned baselines.",
        ("Generator", *(label for _, label in COMPARISON)),
        rows,
        (
            "Models: best.pt scored by digress.cli.evaluate_test. Baselines: digress.cli.evaluate_baselines. TV: "
            "total variation distance to the test molecules (0 = identical distributions, 1 = no overlap).",
        ),
    )
