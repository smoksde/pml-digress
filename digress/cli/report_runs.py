import argparse
from pathlib import Path

from digress.report.style import save
from digress.report.tables import format_table, table_to_latex
from digress.config import BASELINES_DIR
from digress.report.runs import FIGURES, comparison_table, load_baselines, load_run, summary_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--baselines", type=Path, default=BASELINES_DIR / "baseline_results.json",
                        help="baseline_results.json from digress.cli.evaluate_baselines (skipped if missing)")
    args = parser.parse_args()
    runs = [load_run(p) for p in args.runs]
    baselines = load_baselines(args.baselines)
    out = Path(args.out)
    for make in FIGURES:
        fig = make(runs)
        if fig is not None:
            save(fig, out / "figures" / f"{make.__name__}.pdf")
    (out / "tables").mkdir(parents=True, exist_ok=True)
    for table in (summary_table(runs, baselines), comparison_table(runs, baselines)):
        if table is None:
            continue
        (out / "tables" / f"{table.name}.tex").write_text(table_to_latex(table, prefix=""))
        print("\n".join(format_table(table)) + "\n")
    print(f"\nwrote figures and tables to {out}")


if __name__ == "__main__":
    main()
