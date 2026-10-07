"""
Pretty console reporting helpers.

Goal: log (.log / stdout) output that can be copied & pasted directly into a
spreadsheet or a document, without changing any CSV/JSON artifacts.

Every report consists of two views of the same data:
  1) A human-readable aligned table (for reading in the terminal/log)
  2) A fenced TSV block (tab-separated; paste straight into Excel/Sheets)

Usage:
    from utils.report import ReportBlock

    rb = ReportBlock(title="Benchmark Summary")
    rb.add_row(metric="CWO", value=0.8123, higher_is_better=True)
    rb.print()
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence


def _fmt_value(v: Any) -> str:
    """Format a single cell value: floats to 4 decimals, NaN -> '-', None -> '-'."""
    if v is None:
        return "-"
    if isinstance(v, float):
        if math.isnan(v):
            return "-"
        return f"{v:.4f}"
    return str(v)


class ReportBlock:
    """
    Collects rows of named columns and renders:
      - an aligned plain-text table
      - a copy-paste friendly TSV block
    """

    def __init__(
        self,
        title: str,
        columns: Sequence[str],
        aligns: Optional[Sequence[str]] = None,
        width: int = 100,
    ):
        self.title = title
        self.columns = list(columns)
        self.aligns = list(aligns) if aligns else ["left"] * len(columns)
        self.rows: List[List[Any]] = []
        self.width = width
        self.notes: List[str] = []

    # ------------------------------------------------------------------
    def add_row(self, *values: Any, **named_values: Any) -> None:
        """Add a row either positionally or by column name kwargs."""
        if named_values:
            values = tuple(named_values.get(c) for c in self.columns)
        self.rows.append(list(values))

    def add_note(self, note: str) -> None:
        self.notes.append(note)

    # ------------------------------------------------------------------
    def _render_aligned(self) -> str:
        str_rows = [[_fmt_value(v) for v in row] for row in self.rows]
        widths = [len(c) for c in self.columns]
        for r in str_rows:
            for i, cell in enumerate(r):
                widths[i] = max(widths[i], len(cell))
        widths = [min(w, 48) for w in widths]

        def fmt_cell(cell: str, i: int) -> str:
            pad = widths[i] - len(cell)
            return (" " + cell + " " * pad) if self.aligns[i] == "left" else (" " * pad + cell + " ")

        sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
        header = "|" + "|".join(fmt_cell(c, i) for i, c in enumerate(self.columns)) + "|"
        lines = [sep, header, sep]
        for r in str_rows:
            lines.append("|" + "|".join(fmt_cell(c, i) for i, c in enumerate(r)) + "|")
        lines.append(sep)
        return "\n".join(lines)

    def _render_tsv(self) -> str:
        lines = ["\t".join(self.columns)]
        for row in self.rows:
            lines.append("\t".join(_fmt_value(v) for v in row))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def render(self) -> str:
        bar = "=" * self.width
        out: List[str] = ["", bar, f"  {self.title}", bar]
        out.append(self._render_aligned())
        if self.notes:
            out.append("")
            for n in self.notes:
                out.append(f"  * {n}")
        out.append("")
        out.append("[TSV] Copy the block below directly into a spreadsheet:")
        out.append("```tsv")
        out.append(self._render_tsv())
        out.append("```")
        out.append(bar)
        return "\n".join(out)

    def print(self) -> None:
        print(self.render(), flush=True)


def print_kv_summary(
    title: str,
    items: Sequence[tuple],
    width: int = 78,
    notes: Optional[List[str]] = None,
) -> None:
    """
    Print a key/value style summary section with a matching TSV block.

    items: sequence of (label, value[, direction]) where direction is 'up'/'down'/None
           ('↑ better' / '↓ better' annotation).
    """
    def _dir_tag(d: Optional[str]) -> str:
        if d == "up":
            return " ↑"
        if d == "down":
            return " ↓"
        return ""

    bar = "=" * width
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)
    label_w = max(len(lbl + _dir_tag(d)) for lbl, _, *rest in
                  [(i[0], i[1], *(i[2:3] if len(i) > 2 else [None])) for i in items]) + 1
    for item in items:
        lbl, val = item[0], item[1]
        d = item[2] if len(item) > 2 else None
        print(f"  {lbl + _dir_tag(d):<{label_w}} : {_fmt_value(val)}")
    if notes:
        print()
        for n in notes:
            print(f"  * {n}")

    # TSV block
    print()
    print("[TSV] Copy the block below directly into a spreadsheet:")
    print("```tsv")
    print("\t".join(["metric", "value"]))
    for item in items:
        lbl, val = item[0], item[1]
        d = item[2] if len(item) > 2 else None
        print(f"{lbl + _dir_tag(d)}\t{_fmt_value(val)}")
    print("```")
    print(bar, flush=True)


def print_section_header(title: str, width: int = 85) -> None:
    bar = "=" * width
    print(f"\n{bar}\n  {title}\n{bar}", flush=True)
