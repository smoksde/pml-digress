import argparse
from pathlib import Path

from digress.report.style import save
from digress.report.tables import format_table, table_to_latex
from digress.report.runs import FIGURES, load_run, summary_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    runs = [load_run(p) for p in args.runs]
    out = Path(args.out)
    for make in FIGURES:
        fig = make(runs)
        if fig is not None:
            save(fig, out / "figures" / f"{make.__name__}.pdf")
    table = summary_table(runs)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "tables" / f"{table.name}.tex").write_text(table_to_latex(table, prefix=""))
    print("\n".join(format_table(table)))
    print(f"\nwrote figures and tables to {out}")


if __name__ == "__main__":
    main()
