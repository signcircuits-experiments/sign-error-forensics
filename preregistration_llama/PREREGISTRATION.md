# PRE-REGISTRATION — Llama-3.3-70B-Instruct held-out validation (sign-flip circuit), det_4x4 only

Dated: 2026-08-29. Approved by: the PI (name withheld for double-blind review).

**Model.** `meta-llama/Llama-3.3-70B-Instruct` (80 layers) — the exact checkpoint
all discovery measurements and frozen targets were made with. Never Llama-3.1,
never the base model.

**Scope.** det_4x4 error + correct domains only. IBP held-out data is not ready;
IBP gets its own future pre-registration and is excluded here in full.

**Rules.** This file is committed **before** any held-out measurement (commit 1);
raw results are committed afterwards (commit 2); history is never rewritten. The
held-out data is touched **once** — a technical crash may be restarted, a disliked
result may not. All numbers below are FINAL at commit. Verdict vocabulary:
CONFIRMED / NEAR-MISS / NULL / MIXED / FAILED-TECHNICAL, judged only against the
table below.

## 1. Data (frozen)

| File | Received | Kept | Split (− / +) |
|---|---|---|---|
| `llama_det_4x4_error_heldout_n192_experiment_ready.csv` | 192 | 192 | 124 / 68 |
| `llama_det_4x4_correct_heldout_n130_experiment_ready.csv` | 132 | 130 | 76 / 54 |

Errors: no drops (192 received = 192 kept). Corrects: 132 candidates
(87 patternA-relocated + 45 patternB); 2 dropped for
`NO_CANDIDATE_KEPT_ORIGINAL` during relocation (no valid alternate sign
found), before any measurement; archived with reasons. All 322 rows verified: char at
`sign_char_offset` = written sign (`wrong_sign` column, label-inversion convention,
corrects included); `prefix_len` = start of assistant turn; `full_input` = original
Llama-3 header-template prompt (`<|begin_of_text|><|start_header_id|>…` —
discovery files verified same format; teacher forcing is exact).

## 2. Frozen instruments (nothing tuned on held-out data)

| Domain | Habit | Check | Controls |
|---|---|---|---|
| det_4x4_error | 77, 79 | 72 | 30, 50 |
| det_4x4_correct | 77, 78 | 73 | 30, 50 |

Unlike Qwen, Llama's correct domain does **not** inherit the error domain's
layers — each has its own frozen `targets.json` (`human_reviewed: true`, from
discovery data only: n=16 errors, n=80 corrects). d-hat: space-prefixed,
norm-weighted. Donor and mean-ablation means computed per current dataset by
the stage (documented, not tuned). s03 never runs on this data.

Site note: s07r and s09 armA were run in discovery at the **error domain's**
sites (habit L77/L79, check L72, control L30) — the held-out run repeats those
exact sites, as named per row below. s08 (c2 on corrects) is the exception: the
stage loads the **running domain's** targets (`s08_steering.py`), so it steers
at the correct domain's own habit layers L77/L78, in discovery and held-out
alike. The correct domain's targets (habit 77/78, check 73) also govern its
capture-style stages (s01/s02/s04).

## 3. Criteria (the promise)

All arms teacher-forced; **no free-text generation anywhere** (launch flags in the
pre-flight report). Anchors are Llama's **own** measured discovery outcomes
(n=16 errors / 15 late-circuit / 80 corrects) — never Qwen's numbers and never
config constants. Three arms whose discovery outcome contradicted its own
prediction get no bar (marked below) — a bar discovery itself fails is never
pre-registered.

| Stage | Test | Discovery anchor | Pass bar (held-out) | Predicted |
|---|---|---|---|---|
| s01 | Logit lens, errors | 13/16 (81%) peak_layer ≥ L60 | ≥ 60% of cases `peak_layer` ≥ 60 | pass |
| s01 | Logit lens, corrects | no final-layer inversion — mean diff rises monotonically to L79 (peak_layer = 79 in 72/80; 0/80 negative at final layer) | descriptive only; monotone late rise replicates in direction (no inversion expected — unlike Qwen) | pass |
| s02 | DLA | pooled mean MLP DLA: L77 +0.91, L79 +3.06 (minus: L77 +2.81; plus: L77 −0.56, L79 +3.47) | pooled mean MLP DLA positive at L77 AND L79; per-sign ordering qualitative (no per-sign bar — discovery plus at L77 is negative) | pass |
| s04 | Projections onto d-hat | minus-written mean > plus-written at L77/L78/L79; pairwise separation 75% / 70% / 100% | minus-written mean projection > plus-written at L77, L78, L79; pairwise separation ≥ 70% at L79 only (L77/L78 descriptive — discovery sits at the line) | pass |
| s05 | Habit-site MLP ablations | L77 −1.50, L79 −2.64; controls +0.24 / +0.25; L72 −0.04 | L77 and L79 mean Δld negative AND below both controls; controls \|mean\| ≤ 0.5. **L72: no bar** (near-zero in discovery; its dedicated test is s09 armA) | pass |
| s05 | Joint 5-site ablation | set [61, 69, 77, 78, 79]; mean −4.87; 9/16 (56%) beat own null p5 (40 null sets); anchor JSON verified byte-identical to the released copy, no invalid marker | same set [61, 69, 77, 78, 79] — deterministic from frozen targets (habit layers + top late-\|DLA\|), identical to discovery, not tuned; mean Δld ≤ −1.0 AND ≥ 15% beat own null p5 | pass |
| s06 | Direction subtraction | α{0.5,1,2}: L77 −0.23/−0.44/−0.99; L79 −0.81/−1.57/−3.11; both monotone; random ctrl −0.02 | pooled (all late errors) mean Δld negative at every α (0.5/1/2/3), monotone, at L77 AND L79; random ctrl ≤ 0.5. α=3 has no discovery anchor (stronger monotonicity test, not a replication point) | pass |
| s07a | Sign-matched patch into errors | L79 −1.60 (fixed 1/15 = 6.7%); L77 **+0.24**; L72 0.00; controls ≈ 0 | L79 mean Δld ≤ −0.8; controls ≤ 0.5. **No fix-rate bar** (discovery itself is 6.7% > 5%); L77 and L72 descriptive (discovery near-null / positive) | pass |
| s07r | Error-mean patch into corrects (n=80) | minus: L77 −1.28, L79 −3.96 (broken 0); plus: L77 −0.86, L79 −1.41 (broken 0); L30 +0.35 / +0.01 | mean Δld negative at L77 AND L79 for both signs; broken ≤ 5%; \|L30\| ≤ 0.5 | pass |
| s08 | c2 on corrects | minus α2 −17.9 (broken 1/62); plus α2 **−6.68** (broken 0/18); CTRL_MAG ≤ 0.22; α3 broke 57/62 minus | minus ≤ −1.0 at α=2; CTRL_MAG ≤ 0.5. **Plus half descriptive only** — discovery contradicted its own "plus unchanged" prediction, so no plus bar | pass |
| s09 | Judge armA (check L72) | minus corrects mean(A1−A2) +0.09 (34/62 positive); plus −0.28 (5/18 positive); broken 1/80 (one plus-written case, L72, in the A2 self-mean control arm — not the error-mean arm) | direction only: minus-written corrects mean(A1−A2) > 0 AND plus-written mean(A1−A2) < 0 at L72; broken ≤ 5% at every site. Flagged: weak signal, and the L30 control gap (+0.43) exceeds L72's — L30 reported alongside, descriptively | pass |
| s09 | Judge armB | 0/15 fixed at every site (L72, L30) at every γ (1.5/2/3); L72 means ≈ 0 | fixes ≤ 5% at every site at every γ ∈ {1.5, 2, 3} — γ=3 included (discovery never breached, unlike Qwen) | **null** (repeated null = confirmed null) |

## 4. Exclusions (pre-registered)

- s00, s03 (re-derivation destroys the design); **all generation experiments**.
- **All IBP arms** (held-out data not ready; separate future pre-registration).
- **s07b cross-sign flip patch** and **s07c error→error 2×2** (no Llama
  discovery arm exists — nothing to anchor).
- **c2 on errors** (no Llama discovery file exists — Phase 2 exploratory only).
- Any analysis whose composition cell is empty.

## 5. Small-n

Discovery anchors rest on n=16 errors / 15 late-circuit — every anchor is
small-n and the bars above are set accordingly (direction + coarse magnitude,
not tight point replication). Held-out per-sign cells (124− / 68+ errors,
76− / 54+ corrects) are all ≥ 50; any derived cell that falls below 50 is
reported descriptively with exact counts (x/n) alongside the point estimate.

## 6. Phase 2 (EXPLORATORY, only after Phase-1 verdicts lock)

The held-out set may then be re-mined as a larger discovery set (layer scan,
subtype/sign slices, c2 on errors, head-level DLA screen, s07b/s07c analogues,
final-layer inversion on corrects). Everything Phase-2 is labeled EXPLORATORY
in the paper and needs its own future held-out set.

## 7. Run mapping

- Pipeline `pipeline_release`: s01, s02, s04, s05, s06, s07, s08, s09.
  Never s00/s03.
- Config: new `MODEL_CONFIGS` entry `llama_heldout` (exact copy of `llama`,
  `results_name: "Llama_heldout"`); data converted to
  `llama_heldout_det_4x4_{error,correct}_experiment_ready.xlsx`, byte-equivalent
  to the frozen CSVs, checksummed in the pre-flight report.
- Env: `SIGN_DATA_DIR` / `SIGN_RESULTS_DIR` → held-out dirs; results archived
  under `heldout/Llama/results/det_4x4_{error,correct}/`.
- Targets: `det_4x4_error/targets.json` and `det_4x4_correct/targets.json`
  copied unmodified to `{results_root}/{domain}/targets.json`.
- Pre-flight: per-case sign-token ids verified — Llama discovery had **two
  token modes** (space-prefixed 482/489 and bare 10/12), so every held-out row's
  mode is checked explicitly; d-hat construction and hook capture demonstrated
  on 2–3 sample cases before the battery.
