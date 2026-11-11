import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SERIES = (BLUE, ORANGE, AQUA)
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#c9c8c2", "#e6e5e1"
FULL_WIDTH, HALF_WIDTH = 6.3, 3.1
RC = {
    "font.size": 8.5,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
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
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
}


def figure(width, height, *grid, **kwargs):
    plt.rcParams.update(RC)
    return plt.subplots(*grid, figsize=(width, height), **kwargs)


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
