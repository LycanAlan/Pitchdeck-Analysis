"""Result formatting.

Every measurement script prints and writes through here, so the markdown table,
the float precision and the CSV dialect are defined exactly once. A script that
formats its own table is a script whose numbers cannot be compared with another
script's.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from ..config import settings

__all__ = ["ResultsTable", "ResultsBundle"]

_SCALAR = (str, int, float, bool)
_SLUG = re.compile(r"[^a-z0-9]+")


def _slug(text: str) -> str:
    return _SLUG.sub("-", text.lower()).strip("-") or "table"


class ResultsTable:
    """One named table of result rows."""

    def __init__(self, title: str, rows: list[dict], note: str = ""):
        self.title = title
        self.rows = rows
        self.note_text = note

    @property
    def columns(self) -> list[str]:
        """Derived on read so incrementally-built tables stay correct."""
        return self._columns(self.rows)

    @staticmethod
    def _columns(rows: Iterable[dict]) -> list[str]:
        """Union of keys in first-seen order, minus non-scalar payloads.

        An evaluator summary carries its per-question `records` list alongside
        its numbers so one object can be both a table row and a CSV source. A
        table cannot render that, so it is dropped here rather than at every call
        site that would otherwise have to remember to pop it.
        """
        ordered: list[str] = []
        rejected: set[str] = set()
        for row in rows:
            for key, value in row.items():
                if value is not None and not isinstance(value, _SCALAR):
                    rejected.add(key)
                elif key not in ordered:
                    ordered.append(key)
        return [key for key in ordered if key not in rejected]

    @staticmethod
    def _cell(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value).replace("\n", " ").replace("|", "\\|")

    def _numeric(self, column: str) -> bool:
        values = [row.get(column) for row in self.rows if row.get(column) is not None]
        return bool(values) and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in values
        )

    def to_markdown(self) -> str:
        """A width-aligned GitHub table under a heading, floats at 3dp."""
        columns = self.columns
        caption = f"\n\n_{self.note_text}_" if self.note_text else ""
        if not columns:
            return f"### {self.title}\n\n_no rows_{caption}"

        matrix = [[self._cell(row.get(c)) for c in columns] for row in self.rows]
        widths = [
            max([3, len(col)] + [len(row[i]) for row in matrix])
            for i, col in enumerate(columns)
        ]
        right = [self._numeric(c) for c in columns]

        def line(cells: list[str]) -> str:
            padded = [
                cell.rjust(widths[i]) if right[i] else cell.ljust(widths[i])
                for i, cell in enumerate(cells)
            ]
            return "| " + " | ".join(padded) + " |"

        rule = "| " + " | ".join(
            ("-" * (widths[i] - 1) + ":") if right[i] else (":" + "-" * (widths[i] - 1))
            for i in range(len(columns))
        ) + " |"

        body = "\n".join(line(row) for row in matrix)
        return f"### {self.title}\n\n{line(columns)}\n{rule}\n{body}{caption}"

    def to_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # newline="" is required or the csv module doubles line endings on Windows.
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows({c: row.get(c, "") for c in self.columns} for row in self.rows)
        return path

    def render(self) -> None:
        print(self.to_markdown())
        print()

    show = render


class ResultsBundle:
    """The tables one measurement run produced, written out together."""

    def __init__(self, title: str = "PitchLens results", directory: Path | None = None):
        self.title = title
        self.directory = directory or settings.paths.results
        self.tables: list[ResultsTable] = []

    def add(self, title: str, rows: list[dict]) -> ResultsTable:
        return self.add_table(ResultsTable(title, rows))

    def add_table(self, table: ResultsTable) -> ResultsTable:
        self.tables.append(table)
        return table

    def render(self) -> None:
        for table in self.tables:
            table.render()

    def write(self) -> Path:
        """Write RESULTS.md plus one CSV per table; returns the markdown path."""
        self.directory.mkdir(parents=True, exist_ok=True)
        for table in self.tables:
            table.to_csv(self.directory / f"{_slug(table.title)}.csv")

        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        body = "\n\n".join(table.to_markdown() for table in self.tables)
        path = self.directory / "RESULTS.md"
        path.write_text(f"# {self.title}\n\n_Generated {stamp}_\n\n{body}\n", encoding="utf-8")
        return path
