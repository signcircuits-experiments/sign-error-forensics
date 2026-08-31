"""
exp_ctrl_mag.py — Magnitude-matched control (S1)
=================================================
PRE-REGISTERED: CTRL_RAND is mechanically null because a random direction has
~zero projection onto the actual MLP output — so "nothing is actually subtracted"
even at large alpha. This experiment creates a genuine magnitude-matched control.

Method:
  Pass 1 (capture): clean forward to get h75/h78 at probe_tok.
  Compute m_L = |(h_L · d̂)| — the magnitude of what C2 actually removes at α=1.
  CTRL_MAG: subtract m_L * r̂_L from each layer, where
    r̂_L = orthogonalized random unit vector (r -= (r·d̂)*d̂; r /= ||r||; seed=0).
    Scale m_L by alpha for dose-response.
  Both L75 and L78 hooked simultaneously (mirrors C2).

Controls (additional):
  CTRL_MAG_L75_ONLY: magnitude-matched subtraction at L75 only (mirrors C1).
  CTRL_MAG_L30L50:   magnitude-matched at boring layers (mirrors CTRL_BORING).

PRE-REGISTERED PREDICTIONS:
  CTRL_MAG flips none and |meanΔ| << C2 at every alpha.
  FAILURE: if CTRL_MAG matches C2 flip rate → the C2 specificity claim is dead.
  Record failure branch in results meta["failure_observed"].

Domains: det_4x4_error, det_5x5_error, ibp_error
Output: {RESULTS_DIR}/{model}/{domain}/v2_mean_ablation/exp_ctrl_mag.json
"""

import os, gc, json, shutil, tempfile
import torch
import pandas as pd
from adapters.base_adapter import BaseAdapter
from config import RESULTS_DIR

ALPHAS = [1.0, 2.0, 3.0]

# ── Pipeline parameterization (set by stage from targets.json) ──────────
HL1, HL2 = 75, 78     # habit layers (Qwen defaults; overridden per model)
CTRL1, CTRL2 = 30, 50 # boring control layers
SEED   = 0


def _compute_d_hat(adapter) -> torch.Tensor:
    tok   = adapter.tokenizer
    m_ids = tok.encode(' -', add_special_tokens=False)
    p_ids = tok.encode(' +', add_special_tokens=False)
    W_U   = adapter.model.lm_head.weight.detach().float().cpu()
    norm_w = adapter.model.model.norm.weight.detach().float().cpu()
    d = (W_U[m_ids[-1]] - W_U[p_ids[-1]]) * norm_w
    return (d / d.norm()).cpu()


def _make_ortho_random(hidden_dim: int, d_hat: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Random unit vector orthogonalized against d̂. Fixed for the whole run."""
    torch.manual_seed(seed)
    r = torch.randn(hidden_dim).float()
    r -= torch.dot(r, d_hat) * d_hat   # remove d̂ component
    return r / r.norm()


def _mlp(adapter, L):
    return adapter.model.model.layers[L].mlp


def _make_ctrl_mag_hook(r_hat, m_L, alpha, pt):
    """Subtract m_L * alpha * r̂ from output at probe_tok."""
    def hook(module, inp, output):
        output = output.clone()
        if output.dim() == 3 and output.shape[1] > pt:
            subtract = (alpha * m_L * r_hat.to(output.device)).to(output.dtype)
            output[0, pt, :] -= subtract
        return output
    return hook


def run_exp_ctrl_mag(adapter: BaseAdapter, df: pd.DataFrame,
                     domain: str, out_file: str, expA_data: dict):
    late_ids = {qid for qid, q in expA_data.items() if q.get("peak_layer", 0) > 5}
    df_late  = df[df["id"].astype(str).isin(late_ids)].copy()

    hidden_dim = adapter.model.lm_head.weight.shape[1]
    d_hat      = _compute_d_hat(adapter)
    r_hat      = _make_ortho_random(hidden_dim, d_hat, seed=SEED)

    print(f"\n[expCtrlMag] {domain} | n={len(df_late)}")
    print(f"  d̂·r̂ (must be ~0): {torch.dot(d_hat, r_hat).item():.6f}")

    results = {}

    for _, row in df_late.iterrows():
        qid              = str(row["id"])
        wrong_sign       = str(row["wrong_sign"])
        correct_sign     = str(row["correct_sign"])
        sign_char_offset = int(row["sign_char_offset"])
        prefix_len       = int(row["prefix_len"])
        full_input       = str(row["full_input"])

        if sign_char_offset <= prefix_len: continue
        if full_input[sign_char_offset] != wrong_sign: continue

        try:
            wrong_sign_tok, correct_sign_tok = adapter.get_sign_token_ids(
                wrong_sign, sign_char_offset, full_input)
        except ValueError as e:
            print(f"  ⚠ {qid}: {e}"); continue

        full_ids  = adapter.tokenize(full_input)
        tok_idx   = adapter.char_to_token_idx(sign_char_offset, full_input)
        probe_tok = tok_idx - 1
        if probe_tok < 1: del full_ids; continue

        # ── Pass 1: capture h75, h78 and baseline ─────────────────────────────
        captured = {}
        def make_cap(L, pt):
            def hook(m, i, o):
                if o.dim()==3 and o.shape[1]>pt:
                    captured[L] = o[0, pt, :].detach().float().cpu()
            return hook

        h_caps = [_mlp(adapter, HL1).register_forward_hook(make_cap(HL1, probe_tok)),
                  _mlp(adapter, HL2).register_forward_hook(make_cap(HL2, probe_tok)),
                  _mlp(adapter, CTRL1).register_forward_hook(make_cap(CTRL1, probe_tok)),
                  _mlp(adapter, CTRL2).register_forward_hook(make_cap(CTRL2, probe_tok))]
        try:
            with torch.no_grad():
                out_base = adapter.model(full_ids, use_cache=False)
        finally:
            for h in h_caps: h.remove()

        logits_base = out_base.logits[0, probe_tok, :].float().cpu()
        base_ld     = (logits_base[wrong_sign_tok] - logits_base[correct_sign_tok]).item()
        del out_base

        # Compute per-layer magnitudes
        m_vals = {}
        for L in [HL1, HL2, CTRL1, CTRL2]:
            if L in captured:
                m_vals[L] = abs(torch.dot(captured[L], d_hat).item())

        q_result = {
            "id": qid, "written_sign": wrong_sign, "baseline_ld": round(base_ld, 4),
            "m_L75": round(m_vals.get(HL1, float("nan")), 4),
            "m_L78": round(m_vals.get(HL2, float("nan")), 4),
            "conditions": {}
        }

        def run_edit(handles):
            try:
                with torch.no_grad():
                    out = adapter.model(full_ids, use_cache=False)
            finally:
                for h in handles: h.remove()
            lg = out.logits[0, probe_tok, :].float().cpu()
            ld = (lg[wrong_sign_tok] - lg[correct_sign_tok]).item()
            flipped = base_ld > 0 and ld < 0
            top1c   = int(lg.argmax().item()) == correct_sign_tok
            del out, lg
            return round(ld, 4), round(ld - base_ld, 4), flipped, top1c

        for alpha in ALPHAS:
            # CTRL_MAG (L75+L78, magnitude-matched, orthogonal direction)
            if HL1 in m_vals and HL2 in m_vals:
                h = [_mlp(adapter, HL1).register_forward_hook(
                         _make_ctrl_mag_hook(r_hat, m_vals[HL1], alpha, probe_tok)),
                     _mlp(adapter, HL2).register_forward_hook(
                         _make_ctrl_mag_hook(r_hat, m_vals[HL2], alpha, probe_tok))]
                ld, delta, fl, t1 = run_edit(h)
                q_result["conditions"].setdefault("CTRL_MAG", {})[f"a{alpha}"] = {
                    "edit_ld": ld, "delta_ld": delta, "flipped": fl, "top1_now_correct": t1}

            # CTRL_MAG_L75_ONLY
            if HL1 in m_vals:
                h = [_mlp(adapter, HL1).register_forward_hook(
                         _make_ctrl_mag_hook(r_hat, m_vals[HL1], alpha, probe_tok))]
                ld, delta, fl, t1 = run_edit(h)
                q_result["conditions"].setdefault("CTRL_MAG_L75", {})[f"a{alpha}"] = {
                    "edit_ld": ld, "delta_ld": delta, "flipped": fl, "top1_now_correct": t1}

            # CTRL_MAG boring layers
            if CTRL1 in m_vals and CTRL2 in m_vals:
                h = [_mlp(adapter, CTRL1).register_forward_hook(
                         _make_ctrl_mag_hook(r_hat, m_vals[CTRL1], alpha, probe_tok)),
                     _mlp(adapter, CTRL2).register_forward_hook(
                         _make_ctrl_mag_hook(r_hat, m_vals[CTRL2], alpha, probe_tok))]
                ld, delta, fl, t1 = run_edit(h)
                q_result["conditions"].setdefault("CTRL_MAG_BORING", {})[f"a{alpha}"] = {
                    "edit_ld": ld, "delta_ld": delta, "flipped": fl, "top1_now_correct": t1}

        results[qid] = q_result
        del full_ids, logits_base
        gc.collect(); torch.cuda.empty_cache()

    # Check failure branch
    ctrl_mag_flips = sum(
        any(v.get("flipped") for v in q["conditions"].get("CTRL_MAG", {}).values())
        for q in results.values()
    )
    failure_observed = ctrl_mag_flips > 0
    if failure_observed:
        print(f"  ⚠ FAILURE BRANCH: CTRL_MAG flipped {ctrl_mag_flips} questions — specificity claim at risk")

    meta = {
        "n_questions": len(results), "alphas": ALPHAS,
        "d_hat_recipe": "space-prefixed, norm-weighted",
        "r_hat_seed": SEED, "d_dot_r": float(torch.dot(d_hat, r_hat).item()),
        "prediction": "CTRL_MAG flips none; |meanΔ| << C2 at every alpha",
        "failure_branch": "if CTRL_MAG matches C2 flip rate, C2 specificity claim is dead",
        "failure_observed": failure_observed,
    }
    output = {"meta": meta, "results": results}
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=os.path.dirname(out_file),
                                     suffix=".tmp", delete=False) as f:
        json.dump(output, f, indent=2); tmp = f.name
    shutil.move(tmp, out_file)
    print(f"\n[expCtrlMag] {len(results)} questions → {out_file}")
    return results
