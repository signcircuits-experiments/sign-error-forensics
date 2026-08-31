# Held-out pre-registration — Qwen2.5-72B-Instruct (sign-flip circuit)

Committed **before** any held-out measurement was run. All tests, pass bars, and
exclusions: **`PREREGISTRATION.md`** (the binding document).

## Files

| File | Rows | What it is |
|---|---|---|
| `PREREGISTRATION.md` | — | Binding analysis plan: criteria table, exclusions, run mapping |
| `qwen_det_4x4_error_heldout_n147_experiment_ready.csv` | 147 | 4x4 determinant, wrong-sign responses |
| `qwen_det_4x4_correct_heldout_n141_experiment_ready.csv` | 141 | 4x4 determinant, correct responses |
| `qwen_ibp_error_heldout_n26_experiment_ready.csv` | 26 | Integration-by-parts, wrong-sign responses |
| `qwen_ibp_correct_heldout_n50_experiment_ready.csv` | 50 | Integration-by-parts, correct responses |
| `qwen_det_4x4_error_targets.json` | — | Frozen target layers (det), from discovery data only |
| `qwen_ibp_error_targets.json` | — | Frozen target layers (IBP), from discovery data only |
| `qwen_det_4x4_correct_targets.json` | — | Same det layers, inherited (not re-derived) |
| `qwen_ibp_correct_targets.json` | — | Same IBP layers, inherited (not re-derived) |

## Data counts

| File | Received | Kept |
|---|---|---|
| det errors | 206 | 147 |
| det corrects | 142 | 141 |
| IBP errors | 36 | 26 |
| IBP corrects | 50 | 50 |

Rows were dropped only before any measurement, for two reasons: content-level
overlap with the discovery set, or no extractable sign. Dropped rows are
archived with reasons (outside this repository).

## Targets

All four `targets.json` files are frozen from discovery data only,
`human_reviewed: true`, reviewed by the PI (name withheld for double-blind
review) on 2026-08-18. They are never re-derived on this data.
