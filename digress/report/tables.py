import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Table:
    name: str
    caption: str
    header: tuple
    rows: list
    notes: tuple = ()


def fmt_n(x):
    return f"{x:,}"


def fmt_pct(share):
    return f"{100 * share:.1f}%"


def fmt_count(n, share):
    return f"{fmt_n(n)} ({fmt_pct(share)})"


def format_table(table, indent="  "):
    cells = [table.header, *table.rows]
    widths = [max(len(str(r[i])) for r in cells) for i in range(len(table.header))]
    numeric = [i > 0 and all(str(r[i])[:1] in "0123456789+-(." for r in table.rows) for i in range(len(widths))]
    return [indent + "  ".join(str(c).rjust(w) if numeric[i] else str(c).ljust(w)
                               for i, (c, w) in enumerate(zip(r, widths))) for r in cells]


_ESCAPES = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
            "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", ">": r"\textgreater{}",
            "<": r"\textless{}"}


def escape(text):
    parts = re.split(r"\^(\d)", str(text))  # "a0^3": odd entries become superscripts
    return "".join(f"\\textsuperscript{{{p}}}" if i % 2 else "".join(_ESCAPES.get(c, c) for c in p)
                   for i, p in enumerate(parts))


def table_to_latex(t, prefix="qm9-"):
    lines = ["\\begin{table}[ht]", "\\centering", f"\\begin{{tabular}}{{l{'r' * (len(t.header) - 1)}}}", "\\toprule",
             " & ".join(map(escape, t.header)) + " \\\\", "\\midrule",
             *(" & ".join(map(escape, row)) + " \\\\" for row in t.rows), "\\bottomrule", "\\end{tabular}",
             f"\\caption{{{escape(' '.join((t.caption, *t.notes)))}}}", f"\\label{{tab:{prefix}{t.name}}}", "\\end{table}", ""]
    return "\n".join(lines)


def write_tables(tables, out_dir, prefix="qm9-"):
    out = Path(out_dir) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    for t in tables:
        if t is not None:
            (out / f"{t.name}.tex").write_text(table_to_latex(t, prefix))
