# PRE-REGISTRATION — Llama-3.3-70B-Instruct held-out validation (sign-flip circuit), IBP only

Dated: 2026-08-31. Approved by: the PI (name withheld for double-blind review).

**Model.** `meta-llama/Llama-3.3-70B-Instruct` (80 layers) — the exact checkpoint all
IBP discovery measurements and frozen targets were made with. Never Llama-3.1, never
the base model.

**Scope.** IBP (integration-by-parts) error + correct domains only. The det_4x4
domains were adjudicated under their own pre-registration and are excluded here.

**Rules.** Identical to the det_4x4 pre-registration: this file is committed
**before** any held-out measurement (commit 1); raw results are committed afterwards
(commit 2); history is never rewritten; commit 1 additionally receives a third-party
timestamp (OSF registration, Zenodo DOI, or signed annotated git tag) so the freeze
does not rest on git history alone. The held-out data is touched **once** — a
technical crash may be restarted, a disliked result may not. All numbers below are
FINAL at commit. Verdict vocabulary: CONFIRMED / NEAR-MISS / NULL / MIXED /
FAILED-TECHNICAL, judged only against §4 and the frozen mapping in §5.

## 1. Data (frozen)

| File | Received | Kept | Split (− / +) |
|---|---|---|---|
| `llama_ibp_error_pts_final_heldout_n21.xlsx` | 21 | 21 | 15 / 6 |
| `llama_ibp_correct_pts_final_heldout_n47.xlsx` | 47 | 47 | 1 / 46 |

No drops on either file. Zero question-id overlap with the discovery sets (n=22
errors, n=23 corrects) — verified id-by-id before this freeze.

**Composition disclosure (central to this document).** The discovery correct set is
100% minus-written (all 23 cases carry the '−' of the IBP rule u·v − ∫v du). The
held-out correct set is 46 plus-written / 1 minus-written, at assorted '+' positions.
This is a property of the mined pools, known before this freeze and unchangeable —
**no correct-side discovery anchor transfers**. We therefore split the criteria into
two regimes, both frozen here: (a) **anchored bars** on the error domain, where
discovery composition (15−/7+) matches held-out (15−/6+); and (b) **OOD-DIRECTIONAL
bars** on the correct domain — anchor-free, sign-agnostic predictions of the circuit
hypothesis, stated in §4 before measurement. We register the OOD bars rather than
declaring the correct domain descriptive because plus-written corrects are the one
population that can dissociate the direction's content from written-token identity:
every anchored measurement so far involves minus-written corrects, where "the
circuit tracks the minus direction" and "the circuit tracks the written token" make
the same prediction. On plus-written corrects they come apart. A held-out set that
merely repeated the discovery composition could not test this.

**Provenance.** Error and correct cases are mined by the engineer from a much larger
generation pool (thousands of IBP attempts) by a fixed criterion — sign errors at the
IBP-rule position (errors) and sign-correct completions (corrects) — that predates
all held-out results and is unchanged from discovery mining. The mining and
relocation scripts ship with the released repo. If the engineer's funnel counts
(pool size → candidates → kept) are available before commit 1 they are appended to
this section; otherwise the funnel is described qualitatively and the scripts stand
as the auditable record. Reviewers can re-run the mining criterion on the released
pool to confirm no result-dependent selection was possible.

## 2. Frozen instruments (nothing tuned on held-out data)

| Domain | Habit | Check | Controls |
|---|---|---|---|
| ibp_error | 77, 78 | 76 (descriptive only — see below) | 30, 50 |
| ibp_correct | 77, 78 | 76 (descriptive only — see below) | 30, 50 |

Both domains share one frozen `targets.json` layer set (from discovery data only:
n=22 errors, n=23 corrects; `human_reviewed: true`). d-hat: space-prefixed,
norm-weighted. Donor and mean-ablation means computed per current dataset by the
stage (documented, not tuned). s03 never runs on this data.

**Check layer L76 carries no pass bar anywhere in this document.** The targets file's
own review note records that L76's margin over the L60 alternative (0.234) is driven
by a single discovery case (`ibp_pts_n11_b211`); excluding it collapses the margin to
0.067. A site selected at low confidence cannot anchor a confirmatory bar. All L76
arms (s05 check ablation, s09 armA, s09 crossdomain) are reported descriptively.

**Late-circuit rule (frozen verbatim).** *A held-out error case is "emb-peak" iff its
s01 `peak_layer` < 10; all other cases are "late-circuit". Emb-peak cases are
excluded from the denominators of s05 (habit and joint), s06, s08, and s09 armB;
they count against the s01 bar like any other sub-L60 peak; their ids are published
with the results.* This is the same rule that produced discovery's 19/22 late-circuit
split (emb-peak ids: `ibp_pts_n8_b13`, `ibp_pts_n9_b8`, `ibp_ptc_n16_b39`). The
held-out late-circuit subset is determined by held-out s01 output mechanically —
no case-by-case judgment.

## 3. Small-n (read before §4)

Every held-out per-sign cell is below the det_4x4 pre-registration's ≥ 50 reporting
rule: errors 15− / 6+, corrects 1− / 46+ (46 is the one exception). This is not a
choice — the IBP mining pool is exhausted; these are all the cases that exist.
Consequences, frozen: **all anchored bars are pooled**; every per-sign quantity is
reported descriptively as exact counts (x/n) beside the point estimate; no bar rests
on a cell smaller than 15 except the correct-domain OOD bars, which rest on the
n=46 plus cell; the n=1 minus correct is reported verbatim as a single case and
enters no statistic. Percentage bars are stated with their exact integer thresholds.

## 4. Criteria (the promise)

All arms teacher-forced; **no free-text generation anywhere** (`--skip-generation`
is the pipeline default; launch flags in the pre-flight report). Anchors are Llama's
**own** measured IBP discovery outcomes (n=22 errors / 19 late-circuit / 23
corrects) — never Qwen's numbers, never det_4x4 numbers, never config constants.
Arms whose discovery outcome contradicted their own prediction get no bar (marked)
— a bar discovery itself fails is never pre-registered. Qwen's IBP results serve
as cross-model generality in the paper and are never a substitute for any Llama
cell, including the missing minus-written correct cell.

### 4a. Error domain — anchored bars (n=21; late-circuit subset per §2 rule)

| # | Stage | Test | Discovery anchor | Pass bar (held-out) | Predicted |
|---|---|---|---|---|---|
| B1 | s01 | Logit lens, errors | 15/22 (68%) `peak_layer` ≥ L60 | ≥ 50% of all 21 cases `peak_layer` ≥ 60 — exact: **≥ 11/21** | pass |
| B2 | s02 | DLA, pooled | mean MLP DLA: L77 +1.12, L78 +2.01 (L79 +1.71, L76 −0.16) | pooled mean MLP DLA positive at L77 AND L78. L79/L76 and per-sign ordering descriptive (discovery plus at L78 is −0.86 — no per-sign bar) | pass |
| B3 | s04 | Projections onto d-hat | minus mean > plus mean at L77 (+0.41 vs −0.17, sep 85%) and L78 (+1.07 vs +0.25, sep 83%); L76 sep 44% | minus-written mean projection > plus-written at L77 AND L78; pairwise separation ≥ 65% at L77 AND L78 (over all 15×6 = 90 pairs). L76 descriptive (near-null in discovery) | pass |
| B4 | s05 | Habit-site MLP ablations (late subset) | late-19: L77 −0.73, L78 −1.64; controls L30 **+0.65**, L50 +0.33; L76 +0.33 | L77 AND L78 mean Δld negative AND below both controls; \|control mean\| ≤ **1.0** each. The control tolerance is 1.0, not det's 0.5, because discovery's own L30 late-subset value is +0.65 — a 0.5 bar would fail discovery. L76, L79_push descriptive | pass |
| B5 | s05 | Joint 5-site ablation | set [64, 66, 77, 78, 79]; late-19 mean −5.25, 13/19 (68%) beat own null p5; pooled −4.01, 59% (40 null sets, seeds 0–39) | same set [64, 66, 77, 78, 79] — deterministic from frozen targets, identical to discovery, not tuned; late-subset mean Δld ≤ −1.0 AND ≥ 15% beat own null p5 — exact: ≥ ⌈0.15·n_late⌉ cases (= 4 if n_late = 21); same 40 null sets, seeds 0–39 | pass |
| B6 | s06 | Direction subtraction (late subset) | α 0.5/1/2: L77 −0.30/−0.59/−1.28; L78 −0.94/−1.93/−3.89; both monotone; random ctrl −0.02; L30/L50 ≤ 0.06 | pooled mean Δld negative at every α ∈ {0.5, 1, 2, 3} AND strictly monotone in α, at L77 AND L78; \|random ctrl\| ≤ 0.5; \|L30\|, \|L50\| mean ≤ 0.5 at every α. α=3 has no discovery anchor (stronger monotonicity point, not a replication point) | pass |
| B7 | s08 | C2 steering (late subset) | C2 α1/2/3: −2.73/−6.04/−9.45; flips 3/19, 5/19, 8/19 (42% at α3); CTRL_MAG all variants \|mean\| ≤ 0.08, 0 flips | C2 mean Δld ≤ −1.0 at every α ∈ {1, 2, 3} AND strictly monotone; flip rate at α=3 ≥ 15% — exact: ≥ ⌈0.15·n_late⌉ (= 4 if n_late = 21); every CTRL_MAG variant \|mean\| ≤ 0.5 with flips ≤ 5% (≤ 1 case). C1/C3/C4 descriptive | pass |
| B8 | s09 | Judge armB (boost, late subset) | fixes: L76 0/19, 0/19, 1/19; L30 1/19 at every γ; max rate 5.3% | fixes ≤ 10% at every site (L76, L30) at every γ ∈ {1.5, 2, 3} — exact: ≤ ⌊0.10·n_late⌋ (= 2 if n_late = 21). The tolerance is 10%, not det's 5%, because discovery's own max is 1/19 = 5.3%. Site-agnostic: this bar asserts boosting fixes nothing, not that L76 is "the" check layer | **null** (repeated null = confirmed null) |

No bar (descriptive only), with reasons frozen now:

- **s07a sign-matched patch into errors** — discovery contradicted its own
  prediction ("L77 corrective"): minus-written L77 measured **+1.14**, L78 +0.31,
  fixed 0/12; the 7 plus-written cases were skipped (`no donor for sign=+` — the
  donor pool, `det_4x4_correct`, is minus-only). Reported per-sign x/n.
- **s07 correct-mean patch into errors (expF)** — discovery is sign-heterogeneous
  (minus: L77 +1.14, L78 +0.31; plus: L77 −3.79, L78 −5.27), so the pooled
  statistic is a composition artifact; held-out composition (15−/6+) differs from
  discovery's late split (12−/7+). Reported per-sign x/n.
- **s01 per-sign, s02 per-sign, all L76 arms** — per §2 and §3.

### 4b. Correct domain — OOD-DIRECTIONAL bars (n=47; 46+ / 1−)

These bars have **no discovery anchor** (discovery corrects are minus-written; see
§1). Each is a sign-agnostic prediction of the circuit hypothesis, frozen here, and
each is labeled OOD in the paper. Discovery minus-written values are printed as
context only — they are not anchors and set no thresholds.

| # | Stage | Test | Discovery context (minus-written, n=23) | OOD pass bar (plus-written, n=46) | Predicted |
|---|---|---|---|---|---|
| B9 | s05 | Habit + control ablations on corrects | L77 +0.13, L78 +0.25, L30 +0.08, L50 −0.04 (near-null everywhere) | \|mean Δld\| ≤ 1.0 at L77, L78, L30, L50 — ablating the habit sites must leave corrects intact regardless of written sign (error-specificity null) | **null** |
| B10 | s05 | Joint 5-site ablation on corrects | set [69, 72, 77, 78, 79]; mean **+0.48**; criterion_met false (4/23 beat null p5) | \|mean target Δld\| ≤ 1.0 — repeated null | **null** |
| B11 | s07r | Error-mean patch into corrects | L77 −1.20, L78 −3.19, L30 −0.28; broken 0/23 | mean Δld negative at L77 AND L78; \|L30 mean\| ≤ 0.5; broken ≤ 5% — exact: ≤ 2/46. Prediction: injecting error-site means disrupts a correct regardless of its written sign | pass |
| B12 | s08 | C2 on corrects | α1 −7.73, α2 −19.54 (broken 0/23); α3 broke 23/23; CTRL_MAG \|mean\| ≤ 0.35, 0 broken | at α=2: mean Δld ≤ −1.0 AND broken ≤ 5% (≤ 2/46); \|CTRL_MAG mean\| ≤ 0.5 at every α. This is the specificity dissociation: the frozen direction moves the logit where a magnitude-matched random direction does not. α=3 descriptive (discovery broke every case — breakage is expected, not a bar) | pass |

Descriptive only on the correct domain, reasons frozen now:

- **s01/s02/s04 corrects** — the discovery pattern (peak L79 in 23/23; 0/23
  final-layer negative; projections) is a minus-written pattern; whether plus-written
  corrects reproduce it is exactly the open OOD question. Reported in full,
  compared to discovery, no bar.
- **s06 on corrects** — the hypothesis is genuinely ambiguous for plus-written
  cases, and we freeze both readings now: if subtracting the minus direction
  disrupts plus-written corrects **much less** than discovery's minus-written
  corrects (α1 −2.83/−3.22 at L77/L78), that favors the direction carrying
  minus-*content*; if disruption is comparable, that favors the direction carrying
  sign-*salience* magnitude. Either outcome is informative; neither is a pass/fail.
- **s09 armA and crossdomain** — L76 arms (no L76 bars, §2). Discovery context:
  armA mean(A1−A2) at L76 +1.61 (23/23 positive), L30 −0.41; crossdomain L76
  γ1.5/2/3: −1.67/−3.89/−8.60, L30 ≈ 0. Reported with x/n.
- **The single minus-written held-out correct** — reported verbatim (id, every
  measurement), enters no statistic.

## 5. Verdict mapping (frozen)

Twelve registered bars: B1–B8 (error domain), B9–B12 (correct domain, OOD). Core
causal trio: **B4, B6, B7**. Controls: the control clauses inside B4, B6, B7, B11,
B12. Verdicts, decided mechanically:

- **CONFIRMED**: ≥ 11/12 bars pass, including all of B4, B6, B7, with no control
  clause violated.
- **NEAR-MISS**: 9–10 bars pass and ≥ 2 of {B4, B6, B7} pass.
- **MIXED**: 6–8 bars pass.
- **NULL**: ≤ 5 bars pass, or ≥ 2 of {B4, B6, B7} fail with the wrong sign
  (mean Δld ≥ 0).
- **FAILED-TECHNICAL**: only for pre-flight failures caught **before** unblinding
  (checksum mismatch, sign-token mode mismatch, unrecoverable crash) — never after
  results are seen.

Error-domain (B1–B8) and correct-domain (B9–B12) sub-tallies are reported alongside
the overall verdict. In the paper, Llama's minus-written correct-side causal claims
remain **discovery-only** claims regardless of the held-out outcome here — the
held-out correct set cannot confirm them (§1), and Qwen's corresponding results are
cited as cross-model generality, never as a substitute.

## 6. Exclusions (pre-registered)

- s00, s03 (re-derivation destroys the design), s05b; **all generation arms**.
- det_4x4 domains (already adjudicated under their own pre-registration).
- **s07b cross-sign flip patch** and **s07c error→error 2×2** (no Llama IBP
  discovery arm exists — nothing to anchor).
- Emb-peak cases from causal-stage denominators (frozen rule, §2; ids published).
- Any analysis whose composition cell is empty (e.g. within-domain per-sign
  separation on corrects: 1 minus case ≠ a cell).

## 7. Phase 2 (EXPLORATORY, only after Phase-1 verdicts lock)

The held-out set may then be re-mined as a larger discovery set (layer scan, sign
slices, per-position analyses, s07b/s07c analogues, minus-correct case study).
Everything Phase-2 is labeled EXPLORATORY in the paper and needs its own future
held-out set.

## 8. Run mapping

- Pipeline `pipeline_release`: s01, s02, s04, s05, s06, s07, s08, s09. Never
  s00/s03/s05b. `--skip-generation` is the default.
- Config: `MODEL_CONFIGS` entry `llama_heldout` (exact copy of `llama`,
  `results_name: "Llama_heldout"`); data converted to
  `llama_heldout_ibp_{error,correct}_experiment_ready.xlsx`, byte-equivalent to the
  frozen source files, checksummed in the pre-flight report.
- Env: `SIGN_DATA_DIR` / `SIGN_RESULTS_DIR` → held-out dirs; results archived under
  `heldout/Llama/results/ibp_{error,correct}/`.
- Targets: the frozen `ibp_error_targets.json` and `ibp_correct_targets.json`
  copied unmodified to `{results_root}/{domain}/targets.json`.
- Pre-flight (before the battery, results-blind): per-row verification that the char
  at `sign_char_offset` equals the written sign (label-inversion convention,
  corrects included); `prefix_len` = start of assistant turn; original Llama-3
  header-template prompt; per-case sign-token ids checked against Llama's two token
  modes (space-prefixed vs bare); d-hat construction and hook capture demonstrated
  on 2–3 sample cases. All flags and checksums in the pre-flight report.
