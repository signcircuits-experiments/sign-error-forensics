# Pre-declaration: held-out single-site L78 MLP ablation (Llama det_4x4)

**Date frozen:** 2026-08-30
**Model:** Llama-3.3-70B-Instruct
**Domain:** det_4x4 (held-out errors n=192, held-out corrects n=130)
**Status of this file:** the Context and Declaration sections below are frozen
before any L78 held-out ablation is executed. Only an Outcome section may be
appended after the run; the frozen sections are never edited.

## Context (why L78, and what is already known)

In the Llama discovery run (n=16 errors), single-site MLP mean-ablation gave
(from `discovery_4x4_DET/s05_ablations/llama_error_habit_ablation.json`):

| Site | Role assigned | Discovery mean Δld |
|---|---|---|
| L77 | habit | −1.497 |
| L78 | push (amplifier) | **−1.649** |
| L79 | habit | −2.640 |
| L30 / L50 | controls | +0.244 / +0.245 |

L78's discovery ablation effect rivals L77's, and L78 shows large held-out
MLP DLA (errors +0.84, corrects +3.49 at L78; see the s02 workbook
`heldout_comparison` sheet). Because L78 was assigned the push/amplifier role,
the frozen preregistration carried only L77/L79 into individual held-out
ablation bars. L78 is covered only indirectly: as a member of the joint
5-site set [61, 69, 77, 78, 79] (s05 joint, PASSED) and as a correct-domain
habit target. **No individual held-out L78 ablation exists anywhere.**

Verified no-peeking statement: `heldout_4x4_DET/s05_ablations/det_4x4_error_habit_ablation.json`
contains no L78 field (checked programmatically 2026-08-30), so the bars below
are set blind to any held-out L78 outcome.

Held-out anchors for comparison, already adjudicated (n=192 errors):
L77 −0.432, L79 −1.786, L72 −0.186, controls L30 +0.296 / L50 +0.051.
Note the discovery→held-out attenuation at L77 (−1.50 → −0.43).

## Declaration (frozen bars)

**Test:** single-site MLP mean-ablation of L78, identical recipe to the s05
held-out run, applied to (a) all 192 held-out det_4x4 errors and (b) all 130
held-out det_4x4 corrects. Metric: Δld = ablated_ld − baseline_ld, where ld is
the logit difference toward the written sign (errors) / correct sign (corrects),
exactly as in s05.

**B1 (primary, s05 convention):** on held-out errors, mean Δld(L78) is negative
AND below both existing held-out control means (L30 +0.296, L50 +0.051).
Verdict CONFIRMED/NULL on B1 alone.

**B2 (secondary, magnitude):** mean Δld(L78) ≤ −0.8 on held-out errors (half
the discovery anchor −1.649, same halving convention as the s07a bar).
Reported pass/fail separately; does not override B1. Rationale for keeping B2
secondary: L77 attenuated to 29% of its discovery effect held-out, so a hard
magnitude bar risks a false NULL for a real effect.

**Role-discrimination rule (descriptive, pre-stated, no pass/fail):**
let R = mean Δld(L78) on corrects ÷ mean Δld(L78) on errors (both expected
negative).
- R ≥ 0.5 → consistent with the **amplifier** (push) role: L78 boosts whatever
  sign is already present, hurting corrects about as much as errors.
- R < 0.5 → **habit-like** asymmetry: L78 preferentially supports the wrong
  written sign, and the discovery role assignment ("push") should be revised
  in the paper.
Edge rule: if mean Δld on errors is in (−0.1, +0.1), R is unstable; report both
means and make no role call.

**Exclusions:** same as the frozen preregistration §4 (contamination-flagged and
broken-generation cases excluded before averaging; counts reported).

## Execution mechanics (informational, not part of the frozen bars)

- Pipeline: `pipeline_release` s05 stage with a site override adding L78_mlp,
  run twice (error domain, correct domain), same seed and token-mode handling
  as the adjudicated held-out run.
- Expected outputs: `det_4x4_error_L78_ablation.json` and
  `det_4x4_correct_L78_ablation.json` under `heldout_4x4_DET/s05_ablations/`.
- Post-run: append an Outcome section here with the run log
  (`2026-08_llama_L78_ablation_run.log`), then build a verified workbook.

## Outcome

**Appended 2026-08-31. Frozen sections above are unedited.**

**Execution:** run on pod jbvar8yu586r03 (2× A100 80GB) via a standalone driver
(`run_L78_ablation.py`) that imports `pipeline_release` unchanged and sets the
expD′ site list to `L78_push` only. Inputs: the adjudicated held-out
experiment_ready files and the adjudicated s01 expA JSONs (uploaded verbatim).
Environment: transformers 4.46.3 / torch 2.4.0 (matches reference runs).
Run log: `2026-08_llama_L78_ablation_run.log` (sha256[:16] e3752d4bc0679152).

**Integrity (checked before adjudication):**
- Per-case `baseline_ld` identical to the adjudicated s05 held-out run for all
  192 error cases (0 mismatches), and identical to the s01 logit-lens L79 value
  for all 322 cases (max |diff| 0.0) — same recipe, tokenization, orientation.
- Exclusions per prereg §4: 0 contamination-flagged, 0 broken/skipped.
  All 192 errors and 130 corrects included.
- Group split matches expA: errors 189 late_circuit + 3 embedding_bias;
  corrects 129 + 1.
- Raw outputs sha256[:16]: error 454075137bad8b73, correct b1bf6d5b158d8c65.

**Results (mean Δld = ablated_ld − baseline_ld):**

| Domain | n | mean Δld | sd | negative cases |
|---|---|---|---|---|
| held-out errors | 192 | **−1.3851** | 2.10 | 134/192 |
| held-out corrects | 130 | **−2.4538** | 2.03 | 110/130 |

**B1 (primary): CONFIRMED.** −1.3851 is negative and below both held-out
control means (L30 +0.296, L50 +0.051).

**B2 (secondary, magnitude): PASS.** −1.3851 ≤ −0.8. Attenuation from the
discovery anchor is mild (−1.649 → −1.385; 84% retained, vs 29% at L77).

**Role-discrimination rule:** errors mean −1.3851 is outside (−0.1, +0.1), so
R is stable. R = (−2.4538)/(−1.3851) = **1.77 ≥ 0.5** → consistent with the
**amplifier (push) role**: ablating L78 hurts corrects more than errors. The
discovery role assignment stands; no revision to the paper's role labels.

**Released files (flat per-case format, derived from the raw outputs with
written_sign/sign_subtype joined from s01 expA):**
- `heldout_4x4_DET/s05_ablations/det_4x4_error_L78_ablation.json`
  (192 cases, sha256[:16] eeb8165812caa575)
- `heldout_4x4_DET/s05_ablations/det_4x4_correct_L78_ablation.json`
  (130 cases, sha256[:16] a9a6a23405cdc99e)

Held-out single-site picture after this run (errors, mean Δld):
L77 −0.432, **L78 −1.385**, L79 −1.786, L72 −0.186, controls L30 +0.296 /
L50 +0.051.
