"""
expC_prime.py — Head mean-ablation with self-repair logging
============================================================

PRE-REGISTERED DESIGN (set before any results seen):

  Targets at L77 (from original expB DLA, pre-registered):
    Pushers:    H44 (+0.286, 97.4%), H47 (+0.212, 94.7%), H40 (+0.161, 76.3%)
    Correctors: H41 (−0.325, 7.9%), H11 (−0.323, 13.2%)
    Control:    H23 (≈0 DLA, 57.9% positive)

  Targets at L75 (read from expB_dla_l75_heads.json at runtime):
    Top-3 by |DLA| in each direction for wrong='+' late-circuit questions.
    One control: head closest to 0 DLA.

  Method: MEAN ablation (NOT zero-ablation).
    Mean = average o_proj input slice for head H at probe_tok,
    computed over the 35 det error contexts in a single pre-pass.
    Replacement applied at probe_tok position only (semantically consistent
    across questions: one token before the sign).

  Self-repair matrix:
    After each ablation, compute DLA change for 9 key components
    (MLPs at L66/L71/L72/L74/L75/L77/L78/L79, Attn at L75).
    Δ final logit_diff + Δ DLA vector = self-repair matrix.
    Pre-registered interpretation:
      One-factor account predicts: small Δ final, large DLA redistribution.
      Modular account predicts:    Δ final ≈ head's DLA magnitude.

  Run on: det-35 error + ibp-10 + emb-12 (all in same job).

Output: {RESULTS_DIR}/{model}/{domain}/expC_prime_head_mean_ablation.json
  Per-question: baseline logit_diff, eff_dir, per-target ablated logit_diff,
    self-repair matrix (ΔDLA for 9 components).
"""

import torch
import json
import os
import gc
import pandas as pd
from adapters.base_adapter import BaseAdapter
from config import RESULTS_DIR
from experiments.mean_ablation_utils import (
    compute_head_means, run_baseline_pass, compute_dla_from_captured,
    safe_save, SELF_REPAIR_MLP_LAYERS, SELF_REPAIR_ATTN_LAYERS
)

# ── PRE-REGISTERED TARGETS ────────────────────────────────────────────────────
L77_TARGETS = {
    "H44": {"layer": 77, "head": 44, "role": "pusher",    "prior_DLA": +0.286},
    "H47": {"layer": 77, "head": 47, "role": "pusher",    "prior_DLA": +0.212},
    "H40": {"layer": 77, "head": 40, "role": "pusher",    "prior_DLA": +0.161},
    "H41": {"layer": 77, "head": 41, "role": "corrector", "prior_DLA": -0.325},
    "H11": {"layer": 77, "head": 11, "role": "corrector", "prior_DLA": -0.323},
    "H23": {"layer": 77, "head": 23, "role": "control",   "prior_DLA": -0.000},
}
L75_N_PUSH = 3   # top-N L75 pushers by mean DLA
L75_N_CORR = 3   # top-N L75 correctors by mean DLA
L75_N_CTRL = 1   # control head closest to zero

# ── PRE-REGISTERED SUCCESS CRITERIA ──────────────────────────────────────────
# One-factor account is supported if:
#   For any target: |Δ final logit_diff| < 0.5  AND
#     sum(|Δ DLA| for 9 self-repair components) > |head's prior DLA| × 0.5
# Modular account is supported if:
#   Δ final logit_diff ≈ −head's prior DLA (within ±0.3)


def _get_l75_targets(l75_heads_path: str, error_ids: set) -> dict:
    """Read expB_dla_l75_heads.json and select top L75 targets."""
    if not os.path.exists(l75_heads_path):
        print(f"  [warn] {l75_heads_path} not found — skipping L75 targets")
        return {}

    with open(l75_heads_path) as f:
        data = json.load(f)

    # Average per-head DLA across error questions
    head_vals = {}
    overlap = any(qid in error_ids for qid in data)
    for qid, q in data.items():
        if overlap and qid not in error_ids:
            continue
        l75_dla = q.get("head_dla", {}).get("75", q.get("head_dla_L75", []))
        for h_idx, dla in enumerate(l75_dla):
            head_vals.setdefault(h_idx, []).append(float(dla))

    head_means = {h: sum(vs)/len(vs) for h, vs in head_vals.items() if vs}
    sorted_by_dla = sorted(head_means.items(), key=lambda x: x[1], reverse=True)

    targets = {}
    push_added = 0
    for h, m in sorted_by_dla:
        if push_added < L75_N_PUSH and m > 0.01:
            targets[f"L75_H{h}"] = {"layer": 75, "head": h, "role": "pusher",
                                      "prior_DLA": round(m, 4)}
            push_added += 1

    corr_added = 0
    for h, m in reversed(sorted_by_dla):
        if corr_added < L75_N_CORR and m < -0.01:
            targets[f"L75_H{h}"] = {"layer": 75, "head": h, "role": "corrector",
                                      "prior_DLA": round(m, 4)}
            corr_added += 1

    # Control: head closest to 0
    ctrl_h = min(head_means.items(), key=lambda x: abs(x[1]))
    targets[f"L75_H{ctrl_h[0]}_ctrl"] = {
        "layer": 75, "head": ctrl_h[0], "role": "control",
        "prior_DLA": round(ctrl_h[1], 4)}

    return targets


def run_expC_prime(adapter: BaseAdapter, df: pd.DataFrame,
                   domain: str, out_file: str,
                   expA_data: dict,
                   l75_heads_path: str):
    """Run head mean-ablation expC' with self-repair logging."""

    # Determine error IDs for mean computation (det late-circuit only)
    error_ids = {qid for qid, q in expA_data.items() if q.get("peak_layer", 0) > 5}

    # Load L75 targets
    l75_targets = _get_l75_targets(l75_heads_path, error_ids)
    all_targets  = {**L77_TARGETS, **l75_targets}

    print(f"\n[expC'] {adapter.model_name} | {domain} | {len(df)} questions")
    print(f"[expC'] Targets: {list(all_targets.keys())}")
    print(f"[expC'] Computing mean head activations over {len(error_ids)} error contexts...")

    # Pre-pass: compute mean head activation for L75 and L77
    head_means = compute_head_means(adapter, df, error_ids)
    print(f"[expC'] Means computed for layers: {list(head_means.keys())}")

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

        # ── Baseline pass: capture final hidden state, MLP/attn DLA,
        #    AND per-head o_proj inputs at L75 and L77 (for calibration)
        h_final_c  = {}
        base_mlp_c = {}
        base_attn_c= {}
        base_oproj_inputs = {}   # {L: tensor[n_heads*head_dim]} at probe_tok

        def cap_last_base(m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            if isinstance(h, torch.Tensor) and h.dim()==3 and h.shape[1]>probe_tok:
                h_final_c['h'] = h[0, probe_tok, :].detach().float().cpu()

        def make_mlp_base(LL):
            def hook(m, i, o):
                h = o[0] if isinstance(o, tuple) else o
                if isinstance(h, torch.Tensor) and h.dim()==3 and h.shape[1]>probe_tok:
                    base_mlp_c[LL] = h[0, probe_tok, :].detach().float().cpu()
            return hook

        def make_attn_base(LL):
            def hook(m, i, o):
                if isinstance(o, torch.Tensor) and o.dim()==3 and o.shape[1]>probe_tok:
                    base_attn_c[LL] = o[0, probe_tok, :].detach().float().cpu()
            return hook

        def make_oproj_input_hook(LL):
            def hook(m, inp, out):
                if isinstance(inp, tuple) and inp[0] is not None:
                    x = inp[0]
                    if x.shape[1] > probe_tok:
                        base_oproj_inputs[LL] = x[0, probe_tok, :].detach().float().cpu()
            return hook

        base_handles = [adapter.get_layer_modules()[-1].register_forward_hook(cap_last_base)]
        for LL in SELF_REPAIR_MLP_LAYERS:
            base_handles.append(adapter.get_mlp_module(LL).register_forward_hook(make_mlp_base(LL)))
        for LL in SELF_REPAIR_ATTN_LAYERS:
            base_handles.append(adapter.get_o_proj(LL).register_forward_hook(make_attn_base(LL)))
        for LL in [75, 77]:   # capture o_proj inputs at both target attention layers
            base_handles.append(adapter.get_o_proj(LL).register_forward_hook(make_oproj_input_hook(LL)))

        with torch.no_grad():
            adapter.model.model(full_ids, use_cache=False)
        for hh in base_handles:
            hh.remove()

        if 'h' not in h_final_c:
            print(f"  ⚠ h_final capture failed"); del full_ids; continue

        eff_dir     = adapter.compute_eff_dir(wrong_sign_tok, correct_sign_tok, h_final_c['h'])
        baseline_ld = adapter.compute_logit_diff(h_final_c['h'], wrong_sign_tok, correct_sign_tok)
        baseline_dla = compute_dla_from_captured(base_mlp_c, base_attn_c, eff_dir)
        print(f"  baseline_ld={baseline_ld:+.4f}")

        # ── Per-target ablations ──────────────────────────────────────────────
        ablation_results = {}

        for tname, tinfo in all_targets.items():
            L, H = tinfo["layer"], tinfo["head"]
            if L not in head_means:
                print(f"    {tname}: no mean available — skipping")
                continue

            mean_slice = head_means[L]   # [n_heads * head_dim]
            s = H * adapter.HEAD_DIM
            e = s + adapter.HEAD_DIM
            mean_h = mean_slice[s:e]     # [head_dim]

            abl_ld_c = {}
            abl_mlp_c = {}
            abl_attn_c = {}

            def abl_hook(module, inp, out):
                if isinstance(inp, tuple) and inp[0] is not None:
                    x = inp[0].clone()
                    if x.shape[1] > probe_tok:
                        x[0, probe_tok, s:e] = mean_h.to(x.device)
                    # Recompute o_proj output with patched input
                    result = torch.nn.functional.linear(
                        x, module.weight,
                        module.bias if module.bias is not None else None)
                    if result.shape[1] > probe_tok:
                        abl_ld_c['out'] = result[0, probe_tok, :].detach().float().cpu()
                    return result

            def make_mlp_hook_abl(LL):
                def hook(m, i, o):
                    h = o[0] if isinstance(o, tuple) else o
                    if isinstance(h, torch.Tensor) and h.dim()==3 and h.shape[1]>probe_tok:
                        abl_mlp_c[LL] = h[0, probe_tok, :].detach().float().cpu()
                return hook

            def cap_last_abl(m, i, o):
                h = o[0] if isinstance(o, tuple) else o
                if isinstance(h, torch.Tensor) and h.dim()==3 and h.shape[1]>probe_tok:
                    abl_ld_c['final'] = h[0, probe_tok, :].detach().float().cpu()

            handles = [
                adapter.get_o_proj(L).register_forward_hook(abl_hook),
                adapter.get_layer_modules()[-1].register_forward_hook(cap_last_abl),
            ]
            # L is the attention layer; MLP at same index is a DIFFERENT component.
            # Capture ALL self-repair MLP layers (no exclusion).
            for LL in SELF_REPAIR_MLP_LAYERS:
                handles.append(adapter.get_mlp_module(LL).register_forward_hook(
                    make_mlp_hook_abl(LL)))
            # Attn at L75 (self-repair)
            def cap_attn75(m, i, o):
                if isinstance(o, torch.Tensor) and o.dim()==3 and o.shape[1]>probe_tok:
                    abl_attn_c[75] = o[0, probe_tok, :].detach().float().cpu()
            handles.append(adapter.get_o_proj(75).register_forward_hook(cap_attn75))

            with torch.no_grad():
                adapter.model.model(full_ids, use_cache=False)
            for hh in handles: hh.remove()

            ablated_ld = 0.0
            if 'final' in abl_ld_c:
                ablated_ld = adapter.compute_logit_diff(
                    abl_ld_c['final'], wrong_sign_tok, correct_sign_tok)

            delta_ld = ablated_ld - baseline_ld

            # Per-head DLA via W_o projection (shape: [hidden, head_dim] @ [head_dim] → [hidden])
            # head_DLA = dot(W_o[:, s:e] @ slice_h, eff_dir)
            # Compute for BOTH the mean slice and the baseline slice of this question.
            W_o_h = adapter.get_o_proj(L).weight.detach().float().cpu()[:, s:e]  # [8192, 128]
            head_mean_proj     = W_o_h @ mean_h                                    # [8192]
            dla_mean_vec       = float(torch.dot(head_mean_proj, eff_dir))

            # Baseline slice for this head (captured above)
            if L in base_oproj_inputs:
                baseline_slice_h = base_oproj_inputs[L][s:e]                      # [128]
                head_base_proj   = W_o_h @ baseline_slice_h                        # [8192]
                dla_baseline_h   = float(torch.dot(head_base_proj, eff_dir))
            else:
                dla_baseline_h   = 0.0

            expected_delta = dla_mean_vec - dla_baseline_h

            abl_dla = compute_dla_from_captured(abl_mlp_c, abl_attn_c, eff_dir)
            repair_matrix = {k: round(abl_dla.get(k,0) - baseline_dla.get(k,0), 4)
                             for k in set(list(abl_dla)+list(baseline_dla))}

            print(f"    {tname} (L{L}_H{H}, {tinfo['role']}): "
                  f"δ_ld={delta_ld:+.4f}  "
                  f"repair_sum={sum(abs(v) for v in repair_matrix.values()):.3f}")

            ablation_results[tname] = {
                "layer":             L,
                "head":              H,
                "role":              tinfo["role"],
                "prior_DLA":         tinfo["prior_DLA"],
                "ablated_ld":        round(ablated_ld, 4),
                "delta_ld":          round(delta_ld, 4),
                "dla_mean_vec":      round(dla_mean_vec, 4),
                "expected_delta":    round(expected_delta, 4),
                "dla_mean_vec":      round(dla_mean_vec, 4),
                "dla_baseline_h":    round(dla_baseline_h, 4),
                "baseline_DLA":      baseline_dla,
                "repair_matrix":     repair_matrix,
            }

        # ── Group assignment ──────────────────────────────────────────────────
        expA_q = expA_data.get(qid, {})
        peak_layer = expA_q.get("peak_layer")
        if peak_layer is None:
            group = "unknown"
        elif peak_layer <= 5:
            group = "embedding_bias"
        else:
            group = "late_circuit"

        results[qid] = {
            "id":             qid,
            "domain":         domain,
            "group":          group,
            "peak_layer":     peak_layer,
            "baseline_ld":    round(baseline_ld, 4),
            "is_error_case":  baseline_ld > 0,
            "ablations":      ablation_results,
        }
        safe_save(results, out_file)

        del full_ids; gc.collect(); torch.cuda.empty_cache()

    print(f"\n[expC'] Done. {len(results)} questions saved to {out_file}")
    return results
