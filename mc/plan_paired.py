"""Emit the GitHub Actions matrix for the twelve paired scenarios."""
import argparse
import json
import os
from pathlib import Path

from build_paired import PERIODS, THRESHOLDS


def plan(runs_per_job):
    jobs = []
    for threshold in THRESHOLDS:
        for period in PERIODS:
            for start in range(1, 101, runs_per_job):
                end = min(100, start + runs_per_job - 1)
                jobs.append({
                    "threshold": threshold,
                    "period": period,
                    "start": start,
                    "end": end,
                    "name": f"paired_{threshold}_P{period}_R{start:03d}-{end:03d}",
                })
    return {"include": jobs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_per_job", type=int, choices=range(5, 101), metavar="5–100")
    matrix = json.dumps(plan(parser.parse_args().runs_per_job), separators=(",", ":"))
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with Path(output).open("a", encoding="utf-8") as handle:
            handle.write(f"matrix={matrix}\n")
    else:
        print(matrix)
