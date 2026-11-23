"""Export and compare paired regression fixtures, or merge completed cloud chunks."""
import argparse
import csv
import math
from contextlib import contextmanager
from pathlib import Path

from openpyxl import Workbook, load_workbook

from build_paired import MODES, PERIODS, THRESHOLDS

SHEETS = (
    "Efficiency", "Risk_Avg_Q", "Water_Level_Avg", "Intensity_Center",
    "Location_Lat", "Location_Lon", "Affected_Line_Count", "Max_Node_Risk",
)
REF_COLUMNS = ("Sheet", "Time", "Run_1", "Run_2")
TIMES = tuple(range(240))


@contextmanager
def workbook(folder):
    files = list(folder.glob("MonteCarlo*.xlsx"))
    if len(files) != 1:
        raise ValueError(f"Expected one workbook under {folder}, found {len(files)}")
    book = load_workbook(files[0], read_only=True, data_only=True)
    try:
        if tuple(book.sheetnames) != SHEETS:
            raise ValueError(f"{files[0]}: unexpected sheets {book.sheetnames}")
        yield book
    finally:
        book.close()


def sheet_rows(book, sheet):
    rows = book[sheet].values
    header = next(rows)
    values = list(rows)
    if header[0] != "Time" or tuple(row[0] for row in values) != TIMES:
        raise ValueError(f"{sheet}: invalid time axis")
    return header, values


def regression_rows(book, exact_columns):
    for sheet in SHEETS:
        header, values = sheet_rows(book, sheet)
        if tuple(header[:3]) != REF_COLUMNS[1:] or (exact_columns and len(header) != 3):
            raise ValueError(f"{sheet}: unexpected Run columns")
        for minute, run_1, run_2, *_ in values:
            yield sheet, minute, run_1, run_2


def export_reference(folder, target):
    with workbook(folder) as book, target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(REF_COLUMNS)
        writer.writerows(regression_rows(book, exact_columns=False))


def compare(folder, reference, tolerance):
    maximum = 0.0
    failures = 0
    first_failure = None
    count = 0
    with workbook(folder) as book, reference.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        if tuple(next(reader)) != REF_COLUMNS:
            raise ValueError(f"{reference}: unexpected CSV header")
        for expected, actual in zip(reader, regression_rows(book, exact_columns=True), strict=True):
            sheet, minute, *observed = actual
            expected_sheet, expected_minute, *archived = expected
            if sheet != expected_sheet or minute != int(expected_minute) or len(archived) != 2:
                raise ValueError(f"{reference}: mismatched sheet/time at {sheet}, {minute}")
            for run_id, (value, ref) in enumerate(zip(observed, archived, strict=True), 1):
                if value is None and ref == "":
                    error = 0.0
                elif isinstance(value, (int, float)) and ref:
                    error = abs(value - float(ref))
                    if not math.isfinite(error):
                        error = math.inf
                else:
                    error = math.inf
                count += 1
                maximum = max(maximum, error)
                if error > tolerance:
                    failures += 1
                    first_failure = first_failure or (sheet, minute, run_id, value, ref)
    print(f"{folder}: {count} run values, max abs error {maximum:.17g}, {failures} differences over {tolerance:g}")
    if failures:
        print(f"First difference (sheet, time, run, actual, reference): {first_failure}")
    return failures == 0


def merge_scenario(chunks, destination):
    expected_runs = [f"Run_{i}" for i in range(1, 101)]
    for mode in MODES:
        by_sheet = {sheet: {} for sheet in SHEETS}
        for chunk in chunks:
            with workbook(chunk / mode) as book:
                for sheet, series in by_sheet.items():
                    header, rows = sheet_rows(book, sheet)
                    for run, column in zip(header[1:], list(zip(*rows))[1:], strict=True):
                        if run in series or any(value is None for value in column):
                            raise ValueError(f"{chunk}/{mode}/{sheet}: duplicate or incomplete {run}")
                        series[run] = column
        result = Workbook(write_only=True)
        for sheet, series in by_sheet.items():
            if set(series) != set(expected_runs):
                missing = sorted(set(expected_runs) - series.keys())
                extra = sorted(series.keys() - set(expected_runs))
                raise ValueError(f"{destination}/{mode}/{sheet}: missing {missing}, extra {extra}")
            page = result.create_sheet(sheet)
            page.append(["Time", *expected_runs])
            for row in zip(TIMES, *(series[run] for run in expected_runs), strict=True):
                page.append(row)
        folder = destination / mode
        folder.mkdir(parents=True, exist_ok=True)
        result.save(folder / "MonteCarlo_Results_merged.xlsx")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("export", "compare", "merge"):
        cmd = sub.add_parser(action)
        cmd.add_argument("source", type=Path)
        cmd.add_argument("destination", type=Path)
        if action == "compare":
            cmd.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()
    if args.action == "merge":
        for threshold in THRESHOLDS:
            for period in PERIODS:
                name = f"{threshold}_P{period}"
                chunks = sorted(args.source.glob(f"paired_{name}_*"))
                merge_scenario(chunks, args.destination / threshold / f"P{period}")
        return
    if args.action == "export":
        args.destination.mkdir(parents=True, exist_ok=True)
    passed = True
    for threshold in THRESHOLDS:
        for mode in MODES:
            source = args.source / threshold
            if args.action == "export":
                source /= "P2"
            reference = args.destination / f"paired_{threshold}_P2_{mode}.csv"
            if args.action == "export":
                export_reference(source / mode, reference)
            else:
                passed &= compare(source / mode, reference, args.tolerance)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
