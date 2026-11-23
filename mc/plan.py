"""Build the GitHub Actions job matrix.

variants : comma list of wait models, e.g. "ning:12,ning:30,half"
periods  : comma list, e.g. "2,100" or "all"
chunks   : semicolon list of threshold index ranges per job, e.g. "1-7;8-14;15-21" or "each" (one job per threshold)
"""
import json
import os
import sys

variants, periods, chunks = sys.argv[1], sys.argv[2], sys.argv[3]
periods = [2, 5, 10, 25, 50, 100] if periods.strip() == "all" else [int(p) for p in periods.split(",") if p.strip()]
chunks = [str(i) for i in range(1, 22)] if chunks.strip() == "each" else [c.strip() for c in chunks.split(";") if c.strip()]
include = []
for variant in [v.strip() for v in variants.split(",") if v.strip()]:
    model, _, door = variant.partition(":")
    door = door or "12"
    for period in periods:
        for chunk in chunks:
            include.append({
                "model": model, "door": door, "period": period, "chunk": chunk,
                "name": f"{model}{door if model == 'ning' else ''}_P{period}_t{chunk.replace(',', '_')}",
            })
if len(include) > 256:
    sys.exit(f"{len(include)} jobs exceed the 256-job matrix limit")
matrix = json.dumps({"include": include})
with open(os.environ.get("GITHUB_OUTPUT", "/dev/stdout"), "a") as handle:
    handle.write(f"matrix={matrix}\n")
print(f"{len(include)} jobs", file=sys.stderr)
