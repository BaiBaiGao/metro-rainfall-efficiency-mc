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
