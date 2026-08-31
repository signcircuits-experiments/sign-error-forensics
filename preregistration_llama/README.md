# Held-out pre-registration — Llama-3.3-70B-Instruct (sign-flip circuit), det_4x4 only

Committed **before** any held-out measurement was run. All tests, pass bars, and
exclusions: **`PREREGISTRATION.md`** (the binding document). IBP held-out data is
not ready; IBP is excluded in full and will get its own pre-registration.

## Files

| File | Rows | What it is |
|---|---|---|
| `PREREGISTRATION.md` | — | Binding analysis plan: criteria table, exclusions, run mapping |
| `llama_det_4x4_error_heldout_n192_experiment_ready.csv` | 192 | det_4x4, wrong-sign responses (124 − / 68 +) |
| `llama_det_4x4_correct_heldout_n130_experiment_ready.csv` | 130 | det_4x4, correct responses (76 − / 54 +, label-inversion convention) |
| `llama_det_4x4_error_targets.json` | — | Frozen target layers (habit 77/79, check 72, controls 30/50), from discovery data only |
| `llama_det_4x4_correct_targets.json` | — | Independently frozen (habit 77/78, check 73, controls 30/50) — unlike Qwen, not inherited from the error domain |

## Data counts

| File | Received | Kept |
|---|---|---|
| det_4x4 errors | 192 | 192 |
| det_4x4 corrects | 132 | 130 |

Rows were dropped only before any measurement, for two reasons: content-level
overlap with the discovery set, or no extractable sign. Dropped rows are
archived with reasons (outside this repository).

## Targets

All `targets.json` files are frozen from discovery data only,
`human_reviewed: true`, reviewed by the PI (name withheld for double-blind
review). They are never re-derived on this data.
