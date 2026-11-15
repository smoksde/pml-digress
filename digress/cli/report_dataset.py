import argparse
from types import SimpleNamespace

from digress.data.dataset import get_dataset
from digress.data.scan import scan_qm9
from digress.data.split import check_split
from digress.report.data_quality import export_data_quality
from digress.report.dataset import build_tables, export_all, render_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", help="export LaTeX tables, PDF figures, report.txt and run.json to this folder")
    parser.add_argument("--no-subset", action="store_true", help="skip the training-subset checks")
    args = parser.parse_args()

    scan = scan_qm9()
    results = SimpleNamespace(scan=scan, split=None if args.no_subset else check_split(scan))
    tables = build_tables(results)
    print(render_report(tables))
    if args.out:
        print(f"\nexported {export_all(results, tables, args.out)} files to {args.out}")
        if results.split:
            print("\n" + render_report(export_data_quality(get_dataset(), args.out)))


if __name__ == "__main__":
    main()
