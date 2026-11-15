import argparse
from pathlib import Path

from digress.report.style import save_figures
from digress.report.tables import format_table, write_tables
from digress.config import BASELINES_DIR
from digress.evaluation.denoising import denoising_table
from digress.report.runs import (FIGURES, comparison_table, failure_examples, failure_table, load_baselines,
                                 load_run)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--baselines", type=Path, default=BASELINES_DIR / "baseline_results.json",
                        help="baseline_results.json from digress.cli.evaluate_baselines (skipped if missing)")
    parser.add_argument("--labels", nargs="+", help="display names for the runs, in the same order")
    args = parser.parse_args()
    if args.labels and len(args.labels) != len(args.runs):
        parser.error("give one label per run")
    runs = [load_run(p, label) for p, label in zip(args.runs, args.labels or [None] * len(args.runs))]
    baselines = load_baselines(args.baselines)
    out = Path(args.out)
    save_figures(FIGURES, runs, out)
    tables = [comparison_table(runs, baselines), failure_table(runs)]
    for run in runs:
        failure_examples(run, out / "figures" / f"failures_{run['dir'].name}.pdf")
        if run["test"] and "denoising" in run["test"]:
            table = denoising_table(run["test"]["denoising"])
            table.name, table.caption = f"denoising_{run['dir'].name}", f"{table.caption[:-1]}: {run['name']}."
            tables.append(table)
    write_tables(tables, out, prefix="")
    for table in filter(None, tables):
        print("\n".join(format_table(table)) + "\n")
    print(f"\nwrote figures and tables to {out}")


if __name__ == "__main__":
    main()
