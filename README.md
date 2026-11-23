# Metro network efficiency under extreme rainfall — Monte Carlo recomputation

Cloud recomputation of the ninth-version Monte Carlo experiment (6 return periods × 21 speed-restriction /
suspension threshold pairs × 300 runs, two-step route choice) on GitHub Actions.

## What is here
- `mc/base_ninth_two_step.py` — the ninth-version two-step-route execution script, copied verbatim (Windows original, CRLF).
- `mc/build_cell.py` — generates the script for one job. It changes only: input/output paths, return period,
  number of runs, the selected threshold indices, and the waiting-time model. Seeds follow the original scheme
  `20260208 + period_id × 1,000,000 + threshold_index × 10,000 + run`, so Run_k is the same rainfall scenario as before.
- `mc/checkpoint.py` — checkpointed batches (office-hours pause removed).
- `mc/regression_check.py` — the legacy model must reproduce the archived workbook (P100, 5/10 mm/h, Run_1–4) before any job runs.
- `mc/summarize.py` — per-cell metrics, threshold ranking, waiting-time diagnostics.
- `inputs/` — the seven input files (verified against `inputs/SHA256SUMS`).

## Waiting-time models
| model | waiting time on a boarding edge |
|---|---|
| `legacy` | original code: λ = station flow / 3600 in the pure-limited-service vacation queue. Saturates at the 7200 s cap almost everywhere. |
| `ning:<doors>` | Ning et al. (2019) conversion: λ̃ = (line flow / 3600 / doors per side) × 0.7 s / 30 s; station flow split equally across its lines; same formula, 0.9 threshold and 7200 s cap. |
| `half` | headway / 2 (random arrival), capped at 7200 s. |

## Running
Actions → monte-carlo → Run workflow. Example inputs:
- comparison test: variants `ning:12,ning:30,half`, periods `2,100`, chunks `1-7;8-14;15-21`, runs `20`
- full run: variants `<chosen model>`, periods `all`, chunks `each`, runs `300` (126 jobs)

Results: one artifact per job (workbooks in the original format), plus `summary` with `summary_cells.csv`,
`summary_ranking.csv` and `summary_wait_diag.csv`.

## Model revisions of 2026-11-23 (`--flags v2`, see `mc/model_v2_patch.py`)
- `alight`: directed station access; boarding = walk + wait, alighting = walk only.
- `distance`: riding time = 1.2 x straight-line inter-station distance / 35 km/h (Maglev 440 s); headways only for waiting.
- `noise`: the N(0, 0.3) spatial perturbation is drawn once per location per run (no minute-to-minute redraw).
- `loop`: Line 4 loop timetable rules (from = to station) apply to the whole line.
- Threshold indices 22 and 23 are the submission scenarios 16/35 and 80/100.
With all flags off the generated script reproduces the archived results (regression gate); the directed
implementation reproduces the undirected one exactly when alighting is given the boarding time.
