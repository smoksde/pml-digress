from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SERIES = (BLUE, ORANGE, AQUA)
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#c9c8c2", "#e6e5e1"
FULL_WIDTH, HALF_WIDTH = 6.3, 3.1
RC = {
    "font.size": 7,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK_2,
    "axes.titlelocation": "left",
    "axes.titleweight": "bold",
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.3,
    "lines.markersize": 2.5,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
}


def figure(width, height, *grid, **kwargs):
    plt.rcParams.update(RC)
    return plt.subplots(*grid, figsize=(width, height), **kwargs)


def save(fig, path, **kwargs):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, **kwargs)
    plt.close(fig)
    return path


def save_figures(makers, data, out_dir):
    figures = [(make.__name__, make(data)) for make in makers]
    return [save(fig, Path(out_dir) / "figures" / f"{name}.pdf") for name, fig in figures if fig is not None]
