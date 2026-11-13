import json
import math
from pathlib import Path

from digress import config
from digress.config import MAX_ATOMS
from digress.data import stats
from digress.data.stats import REASON_LABELS, UNITS
from digress.evaluation.similarity import FP_BITS, FP_RADIUS, THRESHOLDS, similarity_summary
from digress.report.runinfo import run_info
from digress.report.style import BLUE, FULL_WIDTH, HALF_WIDTH, INK, INK_2, MUTED, ORANGE, figure, save
from digress.report.tables import Table, format_table, table_to_latex

SETTINGS = ("MAX_ATOMS", "QM9_SUBSET_SIZE", "TRAIN_FRACTION", "VAL_FRACTION", "RANDOM_SEED", "LABEL_PROPERTY")


def _n(x):
    return f"{x:,}"


def _pct(share):
    return f"{100 * share:.1f}%"


def _count(n, share):
    return f"{_n(n)} ({_pct(share)})"


def scan_tables(scan):
    counts, total = stats.outcome_counts(scan), scan.total
    explicit_h = sum("H" in symbols for symbols in scan.symbols)
    overview = Table("overview", "Outcome of the molecule filter on all QM9 molecules.", ("Outcome", "Molecules", "Share"),
                     [("QM9 molecules in PyG", _n(total), "100.0%")]
                     + [(REASON_LABELS[k], _n(v), _pct(v / total)) for k, v in counts.items()],
                     (f"{_n(explicit_h)} of the skipped molecules ({_pct(explicit_h / total)}) fail only because RDKit kept an "
                      "explicit H (stereo C=N-H imines) when parsing.",))

    rows, above = stats.atom_count_table(scan)
    n_usable = sum(r[2] for r in rows)
    atoms = Table("atom_counts", "Atoms per molecule after parsing.", ("Atoms", "Parsed", "Usable", "Share of usable"),
                  [(n, _n(p), _n(u), _pct(u / n_usable)) for n, p, u in rows] + [(f">{MAX_ATOMS}", _n(above), "0", "-")])

    elements = Table("elements", "Element composition of usable molecules.",
                     ("Element", "Atoms", "Share of atoms", "Share of molecules"),
                     [(el, _n(n), _pct(a), _pct(m)) for el, n, a, m in stats.element_table(scan)])

    bond_rows, per_molecule = stats.bond_table(scan)
    bonds = Table("bonds", "Bond types of usable molecules (Kekule form).", ("Bond", "Count", "Share"),
                  [(n, _n(c), _pct(s)) for n, c, s in bond_rows], (f"{per_molecule:.1f} bonds per molecule.",))
    rings = Table("rings", "Rings per usable molecule.", ("Rings", "Molecules", "Share"),
                  [(k, _n(c), _pct(s)) for k, c, s in stats.ring_table(scan)])

    smiles, flat = stats.usable_values(scan, scan.smiles), stats.usable_values(scan, scan.smiles_flat)
    stereo = sum(any(c in s for c in "@/\\") for s in smiles)
    duplicates = Table("duplicates", "Unique and duplicate usable molecules.", ("Identity", "Unique", "Duplicates"),
                       [("SMILES with stereo", _n(len(set(smiles))), _n(len(smiles) - len(set(smiles)))),
                        ("graphs without stereo", _n(len(set(flat))), _n(len(flat) - len(set(flat))))],
                       (f"{_n(stereo)} usable molecules ({_pct(stereo / len(smiles))}) carry stereo marks.",))

    summaries = [stats.summarize_target(scan, name) for name in UNITS]
    targets = Table("targets", "QM9 regression targets over usable molecules.",
                    ("Target", "Unit", "Mean", "Std", "Min", "Median", "Max", "Skew", "Outliers"),
                    [(s.name, UNITS[s.name], f"{s.mean:.3g}", f"{s.std:.3g}", f"{s.minimum:.3g}", f"{s.median:.3g}",
                      f"{s.maximum:.3g}", f"{s.skew:+.2f}", _count(s.outliers, s.share)) for s in summaries],
                    (f"Outliers: robust z-score |x - median| / (1.4826 MAD) > {stats.ROBUST_Z:g}. Total energies "
                     "(u0, u, h, g) are nearly identical to each other and scale with molecule size, so their tails "
                     "mostly reflect composition, not errors.",))
    return [overview, atoms, elements, bonds, rings, duplicates, targets]


def split_tables(c):
    subset = Table("subset", "Training subset and its split.", ("Quantity", "Value"),
                   [("molecules in subset", _n(c.size)),
                    ("duplicate graphs skipped while sampling", _n(c.skipped_duplicates)), ("train", _n(c.n_train)), ("validation", _n(c.n_val)), ("test", _n(c.n_test)),
                    ("validation graphs also in train", _count(c.leaked_val_graphs, c.leaked_val_graphs / c.n_val)),
                    ("test graphs also in train", _count(c.leaked_test_graphs, c.leaked_test_graphs / c.n_test)),
                    ("atom-count distance to full set (TV)", f"{c.atom_count_distance:.4f}")])

    properties = Table("properties", "QM9 targets: full usable set versus training subset.",
                       ("Property", "Full mean", "Full std", "Subset mean", "Subset std", "Shift (full std)"),
                       [(p.name, f"{p.full_mean:.3f}", f"{p.full_std:.3f}", f"{p.subset_mean:.3f}",
                         f"{p.subset_std:.3f}", f"{p.shift:+.3f}") for p in c.property_shifts])

    test, base = similarity_summary(c.test_similarity), similarity_summary(c.baseline_similarity)
    rows = [("median similarity", f"{test['median']:.2f}", f"{base['median']:.2f}"),
            ("90th percentile", f"{test['q90']:.2f}", f"{base['q90']:.2f}")]
    rows += [("similarity 1.0" if t == 1.0 else f"similarity at least {t}", _count(*test[t]), _count(*base[t]))
             for t in THRESHOLDS]
    near = Table("near_duplicates", "Similarity of each molecule to its nearest train molecule.",
                 ("", "Test vs train", "Train vs rest of train"), rows,
                 (f"Similarity 1.0 means identical fingerprints, not necessarily identical graphs. Morgan fingerprints "
                  f"(radius {FP_RADIUS}, {FP_BITS} bits) of stereo-free graphs, Tanimoto; baseline: "
                  f"{len(c.baseline_similarity):,} train molecules vs the rest of train.",))
    return [subset, properties, near]


def build_tables(results):
    return scan_tables(results.scan) + (split_tables(results.split) if results.split else [])


def render_report(tables):
    lines = []
    for t in tables:
        lines += ["", t.caption, *format_table(t), *(f"  note: {n}" for n in t.notes)]
    return "\n".join(lines).lstrip("\n")


def _bars(ax, labels, values, texts, title, ylabel, offset=0.0, width=0.7, color=BLUE, label=None):
    bars = ax.bar([i + offset for i in range(len(labels))], values, width, color=color, label=label)
    for bar, text in zip(bars, texts):
        if text:
            ax.annotate(text, (bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 3),
                        textcoords="offset points", ha="center", color=INK_2, fontsize=7.5)
    ax.set_xticks(range(len(labels)), [str(l) for l in labels])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)


def filter_funnel(results):
    scan = results.scan
    rows = stats.filter_funnel(scan)[::-1]
    fig, ax = figure(FULL_WIDTH, 2.4)
    labels = [r[0] for r in rows]
    ax.barh(labels, [scan.total] * len(rows), color=MUTED, alpha=0.35, height=0.62)
    bars = ax.barh(labels, [r[1] for r in rows], color=BLUE, height=0.62)
    for bar, (_, remaining, removed) in zip(bars, rows):
        y = bar.get_y() + bar.get_height() / 2
        ax.annotate(f"{remaining:,}  ({100 * remaining / scan.total:.1f}%)", (remaining, y), xytext=(-4, 0),
                    textcoords="offset points", ha="right", va="center", color="white", fontsize=7.5)
        if removed:
            ax.annotate(f"−{removed:,} ({100 * removed / scan.total:.1f}%)", (scan.total, y), xytext=(4, 0),
                        textcoords="offset points", ha="left", va="center", color=INK_2, fontsize=7.5)
    ax.set_xlim(0, scan.total)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_title("QM9 molecules remaining after each filter step (removed by the step on the right)")
    return fig


def atom_counts(results):
    rows, _ = stats.atom_count_table(results.scan)
    shares = [100 * r[2] / sum(r[2] for r in rows) for r in rows]
    fig, ax = figure(HALF_WIDTH, 2.2)
    _bars(ax, [r[0] for r in rows], shares, [f"{s:.1f}" if s >= 1 else "" for s in shares],
          "Molecule size", "share of usable molecules (%)")
    ax.set_xlabel("heavy atoms per molecule")
    return fig


def elements(results):
    rows = stats.element_table(results.scan)
    fig, ax = figure(HALF_WIDTH, 2.2)
    for offset, col, color, label in ((-0.19, 2, BLUE, "share of atoms"), (0.19, 3, ORANGE, "molecules containing it")):
        values = [100 * r[col] for r in rows]
        _bars(ax, [r[0] for r in rows], values, [f"{v:.1f}" for v in values], "Elements (usable molecules)", "%",
              offset, 0.36, color, label)
    ax.set_ylim(0, 112)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=2, borderaxespad=0.2, handlelength=1, columnspacing=1)
    ax.set_title("Elements (usable molecules)", pad=18)
    return fig


def bonds_and_rings(results):
    bond_rows, per_molecule = stats.bond_table(results.scan)
    ring_rows = stats.ring_table(results.scan)
    fig, (a, b) = figure(FULL_WIDTH, 2.1, 1, 2, gridspec_kw={"width_ratios": [1, 2], "wspace": 0.35})
    _bars(a, [r[0] for r in bond_rows], [100 * r[2] for r in bond_rows], [f"{100 * r[2]:.1f}" for r in bond_rows],
          "Bond types", "share of bonds (%)", width=0.6)
    a.set_xlabel(f"{per_molecule:.1f} bonds per molecule")
    _bars(b, [r[0] for r in ring_rows], [100 * r[2] for r in ring_rows],
          [f"{100 * r[2]:.1f}" if r[2] >= 0.01 else "" for r in ring_rows], "Rings", "share of molecules (%)")
    b.set_xlabel("rings per molecule")
    return fig


def target_distributions(results):
    cols = 3
    rows = math.ceil(len(stats.FIGURE_TARGETS) / cols)
    fig, axes = figure(FULL_WIDTH, 1.75 * rows, rows, cols)
    for ax, name in zip(axes.flat, stats.FIGURE_TARGETS):
        s = stats.summarize_target(results.scan, name)
        x = stats.property_values(results.scan, name).numpy()
        ax.hist(x, bins=60, color=BLUE, log=True)
        for fence in (s.low_fence, s.high_fence):
            if x.min() <= fence <= x.max():
                ax.axvline(fence, color=INK, lw=0.8, ls="--")
        ax.set_title(f"{name} [{stats.PRETTY_UNITS[name]}] · {s.outliers:,} outliers", fontsize=7.5)
        ax.tick_params(labelsize=6.5)
    fig.suptitle(f"QM9 targets over usable molecules (log counts; dashed: outlier fences, robust |z| = {stats.ROBUST_Z:g})",
                 x=0.01, ha="left", fontsize=9, fontweight="bold")
    fig.tight_layout()
    return fig


def near_duplicates(results):
    if results.split is None:
        return None
    c = results.split
    fig, ax = figure(HALF_WIDTH * 1.25, 2.3)
    for sims, color, style, label in ((c.test_similarity, BLUE, "-", "test vs train"),
                                      (c.baseline_similarity, ORANGE, (0, (3, 2)), "train vs rest of train")):
        x = sims.sort().values.numpy()
        ax.step(x, [(i + 1) / len(x) for i in range(len(x))], where="post", color=color, lw=1.6, ls=style, label=label)
    for t in THRESHOLDS[:-1]:
        ax.axvline(t, color=MUTED, lw=0.8)
        ax.text(t, 0.04, f" {t}", color=INK_2, fontsize=7)
    ax.text(0.02, 0.62, "curves overlap: test is no\ncloser to train than train\nis to itself", transform=ax.transAxes,
            fontsize=7, color=INK_2, va="top")
    ax.set_xlabel("Tanimoto similarity to nearest train molecule")
    ax.set_ylabel("cumulative share")
    ax.set_xlim(0, 1.02)
    ax.grid(True, axis="y")
    ax.legend(loc="upper left")
    ax.set_title("Nearest-neighbour similarity")
    return fig


FIGURES = (filter_funnel, atom_counts, elements, bonds_and_rings, target_distributions, near_duplicates)


def export_all(results, tables, out_dir):
    out = Path(out_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    for t in tables:
        (out / "tables" / f"{t.name}.tex").write_text(table_to_latex(t))
    n_figures = 0
    for make in FIGURES:
        fig = make(results)
        if fig is not None:
            save(fig, out / "figures" / f"{make.__name__}.pdf")
            n_figures += 1
    (out / "report.txt").write_text(render_report(tables) + "\n")
    settings = {k: getattr(config, k) for k in SETTINGS}
    (out / "run.json").write_text(json.dumps({**run_info(), "settings": settings}, indent=2) + "\n")
    return len(tables) + n_figures + 2
