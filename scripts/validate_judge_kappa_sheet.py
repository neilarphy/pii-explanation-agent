"""Validate the blind Issue #14 annotation sheet using only its public source."""

import argparse
import csv
from pathlib import Path

SHEET_DIR = Path(__file__).resolve().parents[1] / "data/labeling/judge-kappa-v1"
ANNOTATION_COLUMNS = ("faithful", "citations_ok", "recommendation_ok", "verdict_ok", "comment")


def read_sheet(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader, [])
        if not header or len(header) != len(set(header)):
            raise ValueError(f"{path.name}: missing or duplicate column names")
        rows = list(reader)
    if len(rows) != 60 or any(len(row) != len(header) for row in rows):
        raise ValueError(f"{path.name}: expected 60 records matching the header width")
    return header, rows


def validate(source: Path, sheet: Path, *, blank: bool = False, partial: bool = False) -> None:
    header, original_rows = read_sheet(source)
    sheet_header, rows = read_sheet(sheet)
    if header != sheet_header:
        raise ValueError("Column names or order differ from sheet.csv")

    item_index = header.index("item")
    annotation_indices = [header.index(name) for name in ANNOTATION_COLUMNS]
    recommendation_index = header.index("рекомендация")
    source_indices = [i for i in range(len(header)) if i not in annotation_indices]
    ids = [row[item_index] for row in rows]
    if "" in ids or len(set(ids)) != 60:
        raise ValueError("Expected 60 distinct, nonempty item IDs")

    found_unfilled = False
    for original, row in zip(original_rows, rows, strict=True):
        item = original[item_index]
        for index in source_indices:
            if original[index] != row[index]:
                raise ValueError(f"{item}: changed source column {header[index]}")

        *scores, comment = (row[index] for index in annotation_indices)
        if blank:
            if any(scores) or comment:
                raise ValueError(f"{item}: expected empty annotation fields")
            continue
        if partial:
            if not any(scores) and not comment:
                found_unfilled = True
                continue
            if found_unfilled:
                raise ValueError(f"{item}: annotation after an unfilled row")
        if comment.strip() == "не могу оценить" and not any(scores):
            continue

        for name, score in zip(ANNOTATION_COLUMNS[:-1], scores, strict=True):
            allowed = {"0", "1"}
            if name == "recommendation_ok" and not original[recommendation_index].strip():
                allowed = {"n/a"}
            if score not in allowed:
                raise ValueError(f"{item}: {name} must be one of {', '.join(sorted(allowed))}")
        if "0" in scores and not comment.strip():
            raise ValueError(f"{item}: comment required for a score of 0")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blank", action="store_true", help="check the unfilled template")
    args = parser.parse_args()
    try:
        validate(SHEET_DIR / "sheet.csv", SHEET_DIR / "sheet-taisia.csv", blank=args.blank)
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f"{error}\n")
    print("OK: 60 records validated")


if __name__ == "__main__":
    main()
