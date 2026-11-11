import re
from dataclasses import dataclass


@dataclass
class Table:
    name: str
    caption: str
    header: tuple
    rows: list
    notes: tuple = ()


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
