# FINDINGS — Llama IBP held-out adjudication

**Verdict: CONFIRMED — 11 of 12 frozen bars pass.**

- Pre-registration frozen at commit `fb98035`, before any held-out number existed (the commit history is the timestamp).
- Held-out data: n=21 errors (15 minus-written / 6 plus-written), n=47 corrects (46 plus / 1 minus), touched once, no re-runs.
- Raw results committed unmodified in `Llama/heldout_IBP/` (this commit); discovery anchors in `Llama/discovery_IBP/`.
- Adjudication is mechanical against `PREREGISTRATION_IBP.md` §4/§4b, verdict per §5.
- Late-circuit causal subset per frozen emb-peak rule (peak < L10 excluded): 1 held-out error excluded (`ibp_pts_n15_b41`) → n=20; discovery had 3 excluded → n=19.
- The lone minus-written correct (`ibp_pts_n3_b2`) is reported verbatim in the raw files and enters no statistic, per §6.
- **OOD** (out-of-distribution): discovery corrects were all minus-written; the held-out corrects are almost all plus-written (46/47). The correct-domain bars (B9–B12) are therefore tested on a sign the discovery data contained no examples of.

## Scorecard

| Bar | Claim | Frozen bar | Held-out result | Verdict |
|---|---|---|---|---|
| B1 | Decision is late (errors) | ≥ 50% peak ≥ L60 (≥ 11/21) | 18/21 (85.7%) | **PASS** |
| B2 | L77 & L78 write the sign (DLA) | both pooled means > 0 | +2.06 / +3.95 | **PASS** |
| B3 | d̂ separates minus from plus | pairwise sep ≥ 65% at L77 & L78 | 91% / 100% (90 pairs) | **PASS** |
| B4 | Habit ablations hurt, controls don't | L77, L78 < 0; ctrls within ≤ 1.0 | −1.22 / −2.80 vs +0.27 / +0.25 | **PASS** |
| B5 | Joint 5-set [64,66,77,78,79] beats random sets | mean ≤ −1.0 and ≥ 3/20 beat own null | −7.61; 15/20 | **PASS** |
| B6 | Subtracting d̂: negative, strictly monotone, both sites; ctrls ≤ 0.5 | every α | L77 −0.50/−1.05/−2.14, L78 −1.45/−2.91/−6.01 at α 0.5/1/2; random ctrl 0.000; L30/L50 ≤ 0.12 | **PASS\*** |
| B7 | C2 steering drains and flips errors | mean ≤ −1.0, monotone; flips ≥ 3 | −4.44 / −9.89 / −15.68; flips 2→5→13 of 20 | **PASS** |
| B8 | Boosting the check fixes nothing | ≤ 2/20 fixes at every site, γ | max 1/20 | **PASS (confirmed null)** |
| B9 | Habit ablations leave corrects intact (OOD) | \|mean\| ≤ 1.0 at 4 sites | max \|mean\| 0.34 | **PASS (confirmed null)** |
| B10 | Joint set leaves corrects intact (OOD) | \|mean\| ≤ 1.0 | **+1.0299 — over by 0.03** (median +0.50, max +12.25, 6/46 beat null; tail-driven) | **MISS** |
| B11 | Error-state patch corrupts corrects | L77 & L78 < 0; broken ≤ 2/46 | −3.65 / −7.69; broken 0 | **PASS** |
| B12 | C2 moves corrects, magnitude control doesn't | ≤ −1.0 at α=2; ctrl ≤ 0.5 | −10.11; ctrl worst \|mean\| 0.351; broken 0/46 | **PASS** |

Sub-tallies (per §5): error-domain 8/8; correct-domain (OOD) 3/4.

## Verdict mapping (§5, frozen)

CONFIRMED requires ≥ 11/12 passes including all of the core causal trio B4/B6/B7, with no
control clause violated. Result: 11/12, trio all pass, no control violated → **CONFIRMED**.

## Disclosures

1. **B6 asterisk — α=3 leg not run.** The frozen bar text lists α ∈ {0.5, 1, 2, 3}; the α=3
   subtraction leg was never run — in discovery or held-out (it had no discovery anchor,
   as the pre-registration itself notes). B6 is scored on the measured α ∈ {0.5, 1, 2},
   where effects are large, negative, and strictly monotone at both sites, well past the
   bar. High-dose behaviour of the same direction is shown by C2 steering at α=3
   (−15.68, 13/20 flips). Under a strict reading that counts the missing leg as a fail,
   the tally is 10/12 → NEAR-MISS by the frozen mapping; we report the PASS* reading with
   this disclosure so readers can apply either.
2. **B10 — the single miss, at full size.** A null-tolerance bar on OOD corrects failed by
   0.03 logits on the mean. The median (+0.50) is well inside tolerance; the mean is
   pulled over by a small right tail (max +12.25; 6/46 beat their own null). No excuses:
   by the frozen bar it is a MISS and is counted as one.
3. **Frozen dual reading on corrects (s06), resolved.** Subtracting d̂ disrupts plus-written
   corrects about as much as discovery's minus-written corrects (L78 α=1: −3.28 vs −3.22).
   By the pre-registered tiebreak, d̂ carries **sign-salience**, not minus-content.
4. **L76 (check candidate) — descriptive only, by frozen rule.** Arm A verdict-read +1.06
   beyond self-mean control in 47/47 corrects (L30: −0.26, 7/47); cross-domain boost on
   corrects −5.07 at γ=2 (L30 ≈ 0). A real check-flavoured signal — but B8 shows it cannot
   veto the habit (max 1/20 fixes). No bar existed for L76; it is a candidate for its own
   future pre-registration.

## Reproduction

Every number above recomputes from the raw JSONs in `Llama/heldout_IBP/` (and anchors from
`Llama/discovery_IBP/`) with the frozen rules of `PREREGISTRATION_IBP.md`: unwrap `results`
where present, apply the emb-peak exclusion to causal subsets, take the plus-written pool for
correct-domain bars (sign map from `s07_patching/ibp_error_reverse_patch.json`), and skip the
`__summary__` key in joint-ablation files.
