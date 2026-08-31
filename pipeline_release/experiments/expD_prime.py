"""
expD_prime.py — Single-MLP mean-ablation with self-repair logging
=================================================================

PRE-REGISTERED DESIGN:

  Targets (from expB DLA, all pre-registered):
    Pushers:    L66 (+7.2%), L71 (+13.8%), L72 (+5.7%), L74 (+5.9%), L77 (+45.3%)
    Correctors: L75 (−15.2%), L78 (−23.7%), L79 (−13.2%)
    Controls:   L30, L50 (mid-network, low expected DLA)

  Method: MEAN ablation at probe_tok position.
    Mean = average MLP output at probe_tok over 35 det error contexts.
    Applied at probe_tok only (not all positions).

  KEY PRE-REGISTERED TEST:
    Corrector ablations (L75, L78, L79):
      If ablating L78 worsens logit_diff by ≈ +1.24 (its prior DLA magnitude),
        → direct-effect picture holds, no compensation.
      If the worsening is much smaller (< 0.4),
        → downstream compensation is real and quantified.
    This is the "genuinely new cell" — pusher ablations were done; corrector
    ablations have never been run on this dataset.

  Self-repair matrix:
    Same 9 components as expC'.
    Pre-registered interpretation: see expC_prime.py docstring.

  Run on: det-35 + ibp-10 + emb-12.

Output: {RESULTS_DIR}/{model}/{domain}/expD_prime_mlp_mean_ablation.json
"""

import torch
import json
import os
import gc
import pandas as pd
from adapters.base_adapter import BaseAdapter
from config import RESULTS_DIR
from experiments.mean_ablation_utils import (
    compute_mlp_means, run_baseline_pass, compute_dla_from_captured,
    safe_save, SELF_REPAIR_MLP_LAYERS, SELF_REPAIR_ATTN_LAYERS
)

# ── PRE-REGISTERED TARGETS ────────────────────────────────────────────────────
MLP_TARGETS = {
    "L66_push": {"layer": 66, "role": "pusher",    "prior_DLA_pct": +7.2,  "type": "mlp"},
    "L71_push": {"layer": 71, "role": "pusher",    "prior_DLA_pct": +13.8, "type": "mlp"},
    "L72_push": {"layer": 72, "role": "pusher",    "prior_DLA_pct": +5.7,  "type": "mlp"},
    "L74_push": {"layer": 74, "role": "pusher",    "prior_DLA_pct": +5.9,  "type": "mlp"},
    "L77_push": {"layer": 77, "role": "pusher",    "prior_DLA_pct": +45.3, "type": "mlp"},
    "L75_corr": {"layer": 75, "role": "corrector", "prior_DLA_pct": -15.2, "type": "mlp"},
    "L78_corr": {"layer": 78, "role": "corrector", "prior_DLA_pct": -23.7, "type": "mlp"},
    "L79_corr": {"layer": 79, "role": "corrector", "prior_DLA_pct": -13.2, "type": "mlp"},
    "L30_ctrl": {"layer": 30, "role": "control",   "prior_DLA_pct": 0.0,   "type": "mlp"},
    "L50_ctrl": {"layer": 50, "role": "control",   "prior_DLA_pct": 0.1,   "type": "mlp"},
    # ── ATTENTION TARGET (pre-registered) ────────────────────────────────────
    # L75 whole-layer attention: mean DLA = +0.734 logits, 97.4% positive (37/38
    # late-circuit questions). Tied with L71 MLP (+0.772). Mean computed over
    # 38 late-circuit questions (peak_layer > 5), same as all MLP targets.
    # prior_DLA_pct sourced from: dot(attn_L75, eff_dir) / true_logit_diff
    # = +0.734 / mean_true_ld ≈ 13.8% (varies by question; use directionally).
    # Pre-registered prediction: slope ≈ 1 (modular/near-direct-effect), based
    # on H63 (slope=0.83, r=0.58) and H62 (slope=0.81, r=0.62) head-level evidence.
    # If slope << 1 (< 0.5): repair tracks L75/L77 neighbourhood, not L77 alone.
    # If slope ≈ 1: confirms all non-L77 components are unrepaired → L77 is the
    # singular thermostat target.
    "L75_attn": {"layer": 75, "role": "pusher",    "prior_DLA_pct": +13.8, "type": "attn"},
}

# ── PRE-REGISTERED SUCCESS CRITERIA (CORRECTED) ──────────────────────────────
# Under mean-ablation, the expected Δlogit_diff under a pure direct-effect
# (modular) model for question q is:
#   Δld_q ≈ dla_mean_vec_q − DLA_q
# where dla_mean_vec_q = dot(mean_output_vector, eff_dir_q).
# This averages to ≈ 0 across the dataset (by construction of the mean),
# so threshold tests on mean Δ are uninformative.
#
# CORRECTED per-question regression criterion:
#   Fit: observed_delta_q ~ slope × (dla_mean_vec_q − DLA_q) + intercept
#   Modular (direct-effect):  slope ≈ 1.0
#   Full compensation:         slope ≈ 0.0
#   Partial compensation:      0 < slope < 1
#
# Per question, log: dla_mean_vec_q, DLA_q, observed_delta_q
# Post-hoc regression gives the slope. Pre-register: slope > 0.5 = direct
# effect dominates; slope < 0.5 = compensation dominates.
# This is testable regardless of how small mean Δ is.
SLOPE_DIRECT_THRESHOLD = 0.5   # slope > this → direct effect dominates


def run_expD_prime(adapter: BaseAdapter, df: pd.DataFrame,
                   domain: str, out_file: str,
                   expA_data: dict):
    """Run MLP mean-ablation expD' with self-repair logging."""

    error_ids = {qid for qid, q in expA_data.items() if q.get("peak_layer", 0) > 5}
    mlp_layers  = [v["layer"] for v in MLP_TARGETS.values() if v.get("type","mlp")=="mlp"]
    attn_layers = [v["layer"] for v in MLP_TARGETS.values() if v.get("type")=="attn"]

    print(f"\n[expD'] {adapter.model_name} | {domain} | {len(df)} questions")
    print(f"[expD'] Targets: {list(MLP_TARGETS.keys())}")
    print(f"[expD'] Computing MLP means for {mlp_layers} + Attn means for {attn_layers}...")

    mlp_means = compute_mlp_means(adapter, df, error_ids, mlp_layers)

    # Compute whole-layer attention means (o_proj OUTPUT at probe_tok)
    attn_means = {}
    if attn_layers:
        from experiments.mean_ablation_utils import compute_mlp_means as _cmm
        raw_attn = {L: [] for L in attn_layers}

        for _, row in df.iterrows():
            if str(row["id"]) not in error_ids: continue
            wrong_sign = str(row["wrong_sign"])
            sign_char_offset = int(row["sign_char_offset"])
            prefix_len = int(row["prefix_len"])
            full_input = str(row["full_input"])
            if sign_char_offset <= prefix_len: continue
            if full_input[sign_char_offset] != wrong_sign: continue
            full_ids = adapter.tokenize(full_input)
            tok_idx  = adapter.char_to_token_idx(sign_char_offset, full_input)
            probe_tok = tok_idx - 1
            if probe_tok < 1:
                del full_ids; gc.collect(); continue

            captured = {}
            def make_attn_out_hook(LL):
                def hook(m, i, o):
                    # o_proj OUTPUT (after linear, before residual add)
                    if isinstance(o, torch.Tensor) and o.dim()==3 and o.shape[1]>probe_tok:
                        captured[LL] = o[0, probe_tok, :].detach().float().cpu()
                return hook

            handles = [adapter.get_o_proj(L).register_forward_hook(make_attn_out_hook(L))
                       for L in attn_layers]
            with torch.no_grad():
                adapter.model.model(full_ids, use_cache=False)
            for h in handles: h.remove()

            for L in attn_layers:
                if L in captured:
                    raw_attn[L].append(captured[L])
            del full_ids; gc.collect(); torch.cuda.empty_cache()

        for L, vs in raw_attn.items():
            if vs:
                attn_means[L] = torch.stack(vs).mean(dim=0)

    print(f"[expD'] MLP means: {list(mlp_means.keys())}  Attn means: {list(attn_means.keys())}")

    results = {}

    for _, row in df.iterrows():
        qid              = str(row["id"])
        wrong_sign       = str(row["wrong_sign"])
        correct_sign     = str(row["correct_sign"])
        sign_char_offset = int(row["sign_char_offset"])
        prefix_len       = int(row["prefix_len"])
        full_input       = str(row["full_input"])

        print(f"\n  [{qid}]")

        if sign_char_offset <= prefix_len:
            print(f"  CONTAMINATION — skipping"); continue
        if full_input[sign_char_offset] != wrong_sign:
            print(f"  CHAR MISMATCH — skipping"); continue

        try:
            wrong_sign_tok, correct_sign_tok = adapter.get_sign_token_ids(
                wrong_sign, sign_char_offset, full_input)
        except ValueError as e:
            print(f"  ⚠ {e} — skipping"); continue

        full_ids  = adapter.tokenize(full_input)
        tok_idx   = adapter.char_to_token_idx(sign_char_offset, full_input)
        probe_tok = tok_idx - 1
        if probe_tok < 1:
            print(f"  ⚠ probe_tok < 1 — skipping"); del full_ids; continue

        # ── Baseline ─────────────────────────────────────────────────────────
        baseline_ld, base_mlp, base_attn = run_baseline_pass(
            adapter, full_ids, probe_tok, wrong_sign_tok, correct_sign_tok,
            SELF_REPAIR_MLP_LAYERS, SELF_REPAIR_ATTN_LAYERS)

        h_final_c = {}
        def cap_final(m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            if isinstance(h, torch.Tensor) and h.dim()==3 and h.shape[1]>probe_tok:
                h_final_c['h'] = h[0, probe_tok, :].detach().float().cpu()
        hh = adapter.get_layer_modules()[-1].register_forward_hook(cap_final)
        with torch.no_grad(): adapter.model.model(full_ids, use_cache=False)
        hh.remove()
        if 'h' not in h_final_c:
            print(f"  ⚠ h_final capture failed"); del full_ids; continue
        eff_dir = adapter.compute_eff_dir(wrong_sign_tok, correct_sign_tok, h_final_c['h'])
        baseline_dla = compute_dla_from_captured(base_mlp, base_attn, eff_dir)
        print(f"  baseline_ld={baseline_ld:+.4f}")

        # ── Per-target ablations ──────────────────────────────────────────────
        ablation_results = {}

        for tname, tinfo in MLP_TARGETS.items():
            L = tinfo["layer"]
            is_attn = tinfo.get("type") == "attn"

            if is_attn:
                if L not in attn_means:
                    print(f"    {tname}: no attn mean — skipping"); continue
                mean_out = attn_means[L]
            else:
                if L not in mlp_means:
                    continue
                mean_out = mlp_means[L]  # [hidden_dim]
            abl_captures = {}

            def make_mlp_abl_hook(target_L, mean_v):
                def hook(m, i, o):
                    h = o[0] if isinstance(o, tuple) else o
                    if isinstance(h, torch.Tensor) and h.dim()==3 and h.shape[1]>probe_tok:
                        patched = h.clone()
                        patched[0, probe_tok, :] = mean_v.to(patched.device)
                        return (patched,) if isinstance(o, tuple) else patched
                return hook

            def make_repair_hook(LL, store):
                def hook(m, i, o):
                    h = o[0] if isinstance(o, tuple) else o
                    if isinstance(h, torch.Tensor) and h.dim()==3 and h.shape[1]>probe_tok:
                        store[LL] = h[0, probe_tok, :].detach().float().cpu()
                return hook

            def make_attn_hook(store):
                def hook(m, i, o):
                    if isinstance(o, torch.Tensor) and o.dim()==3 and o.shape[1]>probe_tok:
                        store[75] = o[0, probe_tok, :].detach().float().cpu()
                return hook

            abl_mlp_c = {}
            abl_attn_c = {}
            abl_final_c = {}

            def cap_last_abl(m, i, o):
                h = o[0] if isinstance(o, tuple) else o
                if isinstance(h, torch.Tensor) and h.dim()==3 and h.shape[1]>probe_tok:
                    abl_final_c['h'] = h[0, probe_tok, :].detach().float().cpu()

            if is_attn:
                # Whole-layer attention mean-ablation: replace o_proj OUTPUT at probe_tok
                def make_attn_abl_hook(mean_v):
                    def hook(m, i, o):
                        if isinstance(o, torch.Tensor) and o.dim()==3 and o.shape[1]>probe_tok:
                            patched = o.clone()
                            patched[0, probe_tok, :] = mean_v.to(patched.device)
                            return patched
                    return hook
                ablation_hook = adapter.get_o_proj(L).register_forward_hook(
                    make_attn_abl_hook(mean_out))
            else:
                ablation_hook = adapter.get_mlp_module(L).register_forward_hook(
                    make_mlp_abl_hook(L, mean_out))

            handles = [
                ablation_hook,
                adapter.get_layer_modules()[-1].register_forward_hook(cap_last_abl),
            ]
            for LL in SELF_REPAIR_MLP_LAYERS:
                if LL != L:
                    handles.append(adapter.get_mlp_module(LL).register_forward_hook(
                        make_repair_hook(LL, abl_mlp_c)))
            handles.append(adapter.get_o_proj(75).register_forward_hook(
                make_attn_hook(abl_attn_c)))

            # ── Recovery curve: capture residual stream at every layer ≥ L ─────
            # Nearly free — the ablated forward pass already runs these layers.
            # We hook every decoder layer from L onwards and project each hidden
            # state through final-LN + W_U to get logit_diff at each depth.
            # Saves to recovery_curve: {layer_idx: logit_diff} in the output JSON.
            recovery_raw = {}   # {layer_idx: hidden_state_at_probe_tok}

            def make_recovery_hook(layer_idx):
                def hook(m, i, o):
                    h = o[0] if isinstance(o, tuple) else o
                    if isinstance(h, torch.Tensor) and h.dim()==3 and h.shape[1]>probe_tok:
                        recovery_raw[layer_idx] = h[0, probe_tok, :].detach().float().cpu()
                return hook

            # Hook every layer from L+1 to end (capturing the propagation of the ablation)
            all_layers = adapter.get_layer_modules()
            recovery_handles = []
            for layer_idx_r, layer_module in enumerate(all_layers):
                if layer_idx_r > L:  # only layers after the ablated one
                    recovery_handles.append(
                        layer_module.register_forward_hook(make_recovery_hook(layer_idx_r)))

            with torch.no_grad():
                adapter.model.model(full_ids, use_cache=False)
            for hh in handles: hh.remove()
            for hh in recovery_handles: hh.remove()

            # Project each captured hidden state → logit_diff
            recovery_curve = {}
            for layer_idx_r, h_r in recovery_raw.items():
                try:
                    ld_r = adapter.compute_logit_diff(h_r, wrong_sign_tok, correct_sign_tok)
                    recovery_curve[layer_idx_r] = round(float(ld_r), 4)
                except Exception:
                    pass

            ablated_ld = 0.0
            if 'h' in abl_final_c:
                ablated_ld = adapter.compute_logit_diff(
                    abl_final_c['h'], wrong_sign_tok, correct_sign_tok)

            delta_ld = ablated_ld - baseline_ld

            # DLA of the mean vector (needed for calibrated criterion)
            dla_mean_vec = float(torch.dot(mean_out, eff_dir))
            # Per-question expected delta under direct-effect model
            baseline_key = f"attn_L{L}" if is_attn else f"mlp_L{L}"
            baseline_dla_this_L = baseline_dla.get(baseline_key, 0.0)
            expected_delta = dla_mean_vec - baseline_dla_this_L

            abl_dla = compute_dla_from_captured(abl_mlp_c, abl_attn_c, eff_dir)
            # Drop the ablated component's own key from repair matrix.
            # For MLP ablation: drop mlp_L{L}. For attn ablation: drop attn_L{L}.
            drop_key = f"attn_L{L}" if is_attn else f"mlp_L{L}"
            repair_keys = {k for k in set(list(abl_dla)+list(baseline_dla))
                           if k != drop_key}
            repair_matrix = {k: round(abl_dla.get(k,0) - baseline_dla.get(k,0), 4)
                             for k in repair_keys}
            repair_sum = sum(abs(v) for v in repair_matrix.values())

            print(f"    {tname} (L{L}, {tinfo['role']}): "
                  f"δ_ld={delta_ld:+.4f}  expected={expected_delta:+.4f}  "
                  f"repair_sum={repair_sum:.3f}")

            ablation_results[tname] = {
                "layer":            L,
                "role":             tinfo["role"],
                "prior_DLA_pct":    tinfo["prior_DLA_pct"],
                "ablated_ld":       round(ablated_ld, 4),
                "delta_ld":         round(delta_ld, 4),
                "dla_mean_vec":     round(dla_mean_vec, 4),
                "expected_delta":   round(expected_delta, 4),
                "baseline_DLA_this": round(baseline_dla_this_L, 4),
                "baseline_DLA":     baseline_dla,
                "repair_matrix":    repair_matrix,
                "recovery_curve":   recovery_curve,  # {layer_idx: logit_diff} after ablation
            }

        expA_q = expA_data.get(qid, {})
        peak_layer = expA_q.get("peak_layer")
        group = ("embedding_bias" if peak_layer is not None and peak_layer <= 5
                 else "late_circuit" if peak_layer else "unknown")

        results[qid] = {
            "id":           qid,
            "domain":       domain,
            "group":        group,
            "peak_layer":   peak_layer,
            "baseline_ld":  round(baseline_ld, 4),
            "is_error_case": baseline_ld > 0,
            "ablations":    ablation_results,
        }
        safe_save(results, out_file)
        del full_ids; gc.collect(); torch.cuda.empty_cache()

    print(f"\n[expD'] Done. {len(results)} questions saved to {out_file}")
    return results
