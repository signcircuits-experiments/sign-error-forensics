# PRE-REGISTRATION — Qwen2.5-72B-Instruct held-out validation (sign-flip circuit)

**Rules.** This file is committed **before** any held-out measurement (commit 1);
raw results are committed afterwards (commit 2); history is never rewritten. The
held-out data is touched **once** — a technical crash may be restarted, a disliked
result may not. All numbers below are FINAL at commit. Verdict vocabulary:
CONFIRMED / NEAR-MISS / NULL / MIXED / FAILED-TECHNICAL, judged only against the
table below.

## 1. Data (frozen)

| File | Received | Kept | Split (− / +) |
|---|---|---|---|
| `qwen_det_4x4_error_heldout_n147_experiment_ready.csv` | 206 | 147 | 73 / 74 |
| `qwen_det_4x4_correct_heldout_n141_experiment_ready.csv` | 142 | 141 | 123 / 18 |
| `qwen_ibp_error_heldout_n26_experiment_ready.csv` | 36 | 26 | 26 / 0 |
| `qwen_ibp_correct_heldout_n50_experiment_ready.csv` | 50 | 50 | 50 / 0 |

Drops: content-level discovery-set overlap or no extractable sign, all before any
measurement; archived with reasons. All 364 rows verified: char at
`sign_char_offset` = written sign (`wrong_sign` column, label-inversion convention,
corrects included); `prefix_len` = start of assistant turn; `full_input` = original
ChatML prompt (teacher forcing is exact).

## 2. Frozen instruments (nothing tuned on held-out data)

| Domain | Habit | Check | Controls |
|---|---|---|---|
| det_4x4 (error + correct) | 75, 78 | 79 | 30, 50 |
| ibp (error + correct) | 75, 78 | 63 | 30, 50 |

Correct domains inherit their paired error domain; all four `targets.json`
`human_reviewed: true`, from discovery data only. d-hat: space-prefixed,
norm-weighted. Donor and mean-ablation means computed per current dataset by the
stage (documented, not tuned). s03 never runs on this data.

## 3. Criteria (the promise)

All arms teacher-forced; **no free-text generation anywhere** (launch flags in the
pre-flight report). Anchors are measured discovery outcomes (n=59 ablation stages,
n=44 patch/judge arms) — not `config.py` constants, which discovery itself did not
meet (L75 −0.46 @ 17% vs `MLP_PUSHER_CONFIRM_DELTA` −1.0 @ 75%) and which would
guarantee spurious failure.

| Stage | Test | Discovery anchor | Pass bar (held-out) | Predicted |
|---|---|---|---|---|
| s01 | Logit lens, error domains | det 43/59 (73%) peak ≥ L60; IBP 10/10 | ≥ 60% of cases `peak_layer` ≥ 60, per domain | pass |
| s01 | Logit lens, corrects | final-layer inversion seen | descriptive only; inversion replicates in direction | pass |
| s02 | DLA (`--matched --head-layers`) | minus: L75 +1.93, L78 +2.77; plus: L77 +2.24 | mean MLP DLA at L75 and L78 positive per domain; per-sign ordering qualitative | pass |
| s04 | Projections onto d-hat | no saved discovery projection files — prediction stated explicitly here, not anchored | det: minus-written mean projection < plus-written mean projection at L75/L78/L79, and per-case projections separate in that order for ≥ 70% of cases; IBP (all minus-written): mean projection at L63/L75/L78 has the same sign as det's minus-written mean | pass |
| s05 | Habit-site MLP ablations | det: L75 −0.46, L78 −0.55, L79 −0.35; controls +0.01 / +0.02. IBP: L75 −0.150, L78 −0.025, L79 −0.562; controls +0.025 / −0.050 (L78 not below both controls in discovery — no bar) | det only: each habit site mean Δld negative AND below both controls; controls \|mean\| ≤ 0.5. IBP: descriptive (its dedicated test is the L63 row) | pass (det) |
| s05 | Joint 5-site ablation | det: mean −2.08; beats null p5 16/59 (27%); discovery joint set was [67,71,72,74,77]. Held-out joint set is [71,74,75,77,78] — deterministic from the frozen targets (habit layers + top late-\|DLA\|), not tuned. IBP: mean −0.76, 0/10 beat null (fails det bar in discovery — no bar) | det only: mean Δld ≤ −1.0 AND ≥ 15% beat own null p5, on the held-out joint set. IBP: descriptive | pass (det) |
| s05 | IBP check L63 | 10/10 negative, mean −0.167, sd 0.034 | mean MLP DLA at L63 negative AND ≥ 75% of 26 cases individually negative | pass |
| s05 | L77-attn re-test (`--with-heads`) | patch 0/44, mean +0.033; judge 0/44 | fix/flip ≤ 5% AND \|mean Δld\| ≤ 0.5 | **null** (repeated null = confirmed null) |
| s06 | Direction subtraction | det α{0.5,1,2}: L78 −0.61/−1.26/−2.55, monotone; random ctrl +0.01. IBP: L78 −0.61/−1.21/−2.50, monotone; random ctrl +0.01 | minus errors (per domain): mean Δld negative at every α (0.5/1/2/3), monotone; random ctrl ≤ 0.5. α=3 has no discovery anchor (stronger monotonicity test, not a replication point) | pass |
| s07a | Sign-matched patch (det, MLP-only) | L75 −0.39 (0/44), L78 −0.87 (1/44), L79 −1.77 (0/44), controls ≈ 0 | mean Δld negative at L75/L78/L79; L79 ≤ −0.8; fix ≤ 5% per site; controls ≤ 0.5 | pass |
| s07b | Cross-sign flip patch (det) | — (rule fixed here) | at eligible sites (per-sign donor-mean L2 gap > within-sign spread; spread = mean L2 distance of each donor to its own-sign mean, computed before patching): opposite-sign mean more negative than sign-matched | pass at eligible sites; others null |
| s07c | Error→error 2×2 (det) | held-out subtypes: Alternating_Drift 70 (44+ / 26−), Cofactor_Neglect 40 (12+ / 28−); leave-one-out donors | \|sign-factor effect\| ≥ 2 × \|subtype-factor effect\|; effects = main effects from the 2×2 cell means of Δld | pass |
| s08 | c2 on errors | C2 in `exp_combined_ablation.json`, det minus n=14: α1 −1.97 (0/14), α2 −4.38 (2/14), α3 −7.30 (5/14); plus n=30 positive (+0.61/+1.39/+2.41); CTRL_MAG −0.08 (`exp_ctrl_mag.json`). IBP: −2.15/−4.84/−7.95; CTRL_MAG −0.30 | minus-written errors (per domain): mean Δld negative at every α, monotone; plus-written reported descriptively; CTRL_MAG ≤ 0.5 | pass |
| s08 | c2 on corrects | n=47: minus α2 −2.99 (broken 8/36); plus α2 +2.27 (0/11); CTRL −0.06 | minus ≤ −1.0 at α=2; plus ≥ 0 at α=2; CTRL_MAG ≤ 0.5. IBP plus half N/A (no plus-writers) | pass |
| s09 | Judge armA (error-mean vs self-mean patch into corrects, `exp_judge_candidates_armA`) | det L79, n=47: per-case A1−A2 gap +1.48 on minus corrects (36/36 positive), −0.59 on plus corrects (0/11 positive); broken 0/47 at every site (muted judge: responds to error content, never flips) | det: minus-written corrects mean(A1−A2) > 0 AND plus-written corrects mean(A1−A2) < 0 at the check layer; broken ≤ 5% at every site. IBP: no discovery armA exists — direction only (minus corrects A1 > A2 at L63), no numeric bar | pass |
| s09 | Judge armB (logit boosts) | det γ≤2: 0/44 everywhere except L79 1/44 at γ=2; γ=3: L79 5/44 (11.4%). No IBP armB exists in discovery | det only: fixes ≤ 5% at every site at γ ≤ 2. γ=3 EXPLORATORY, no bar (discovery itself breached 5% there). IBP: descriptive | pass (det) |

## 4. Exclusions (pre-registered)

- s00, s03 (re-derivation destroys the design); **all generation
  experiments** (retracted; s08 runs c2 arms only, s09 logit arms only).
- L77-attn patching (no attention path in s07; covered by the s05 re-test).
- IBP: entire s07 suite (zero plus-writers; no discovery baseline).
- Any analysis whose composition cell is empty.

## 5. Small-n

Every test with n < 50 (all IBP arms, plus-written det corrects n=18, 2×2 cells)
is flagged as small-n and reported descriptively with exact counts (x/n)
alongside the point estimate.

## 6. Phase 2 (EXPLORATORY, only after Phase-1 verdicts lock)

The held-out set may then be re-mined as a larger discovery set (layer scan,
subtype/sign slices, head-level DLA screen from s02, s05b if the screen hits,
final-layer inversion on corrects, a weight-space check of the habit layers,
a prompt-surface-feature scan of the L0 lean). Everything Phase-2 is labeled
EXPLORATORY in the paper and needs its own future held-out set.

## 7. Run mapping

- Pipeline `pipeline_release`: s01, s02, s04, s05, s06, s07, s08, s09.
  Never s00/s03.
- Config: new `MODEL_CONFIGS` entry `qwen_heldout` (exact copy of `qwen`,
  `results_name: "Qwen_heldout"`); data converted to
  `qwen_heldout_{domain}_experiment_ready.xlsx`, byte-equivalent to the frozen
  CSVs, checksummed in the pre-flight report.
- Env: `SIGN_DATA_DIR` / `SIGN_RESULTS_DIR` → held-out dirs; results archived
  under `heldout/Qwen/results/{domain}/v1_zero_ablation|v2_mean_ablation/`.
- Targets: each `targets.json` copied unmodified to
  `{results_root}/{domain}/targets.json`.
- Pre-flight: per-case sign-token ids verified (bare vs space-prefixed — discovery
  det had 2/59 bare); d-hat construction and hook capture demonstrated on 2–3
  sample cases before the battery.
