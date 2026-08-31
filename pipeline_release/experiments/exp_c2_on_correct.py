"""
exp_c2_on_correct.py — Bidirectional test: C2 on correct answers (S3)
=======================================================================
PRE-REGISTERED: applying C2 (d̂ subtraction at L75.mlp + L78.mlp) to correct
answers should weaken '−'-correct cases (their correct answers become less
confident) but leave '+'-correct cases unchanged or slightly improved.

⚠ LABEL INVERSION: in det_4x4_correct, written sign = row["wrong_sign"].

Method: same C2 hook stack as exp_combined_ablation.
Alphas: {1.0, 2.0, 3.0}. CTRL_MAG included (from exp_ctrl_mag pattern).
No late-circuit filter — all 47 correct questions included (record peak_layer if expA available).

PRE-REGISTERED PREDICTIONS:
  '−'-written corrects (n=36): ld drops toward '+' (WEAKENED). Margins are
    +10..20; expect "worsens, rarely breaks" — breaks are bonus, not headline.
  '+'-written corrects (n=11): ld unchanged or slight improvement.
  CTRL_MAG: null across both groups.

Output: {RESULTS_DIR}/{model}/det_4x4_correct/v2_mean_ablation/exp_c2_on_correct.json
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


def _make_ortho_random(hidden_dim, d_hat, seed=0):
    torch.manual_seed(seed)
    r = torch.randn(hidden_dim).float()
    r -= torch.dot(r, d_hat) * d_hat
    return r / r.norm()


def _mlp(adapter, L):
    return adapter.model.model.layers[L].mlp


def _make_sub_hook(direction, alpha, pt):
    def hook(m, i, o):
        out = o.clone()
        if out.dim() == 3 and out.shape[1] > pt:
            h    = out[0, pt, :].float()
            proj = torch.dot(h, direction.to(h.device))
            out[0, pt, :] -= (alpha * proj * direction.to(h.device)).to(out.dtype)
        return out
    return hook


def _make_ctrl_mag_hook(r_hat, m_L, alpha, pt):
    def hook(m, i, o):
        out = o.clone()
        if out.dim() == 3 and out.shape[1] > pt:
            out[0, pt, :] -= (alpha * m_L * r_hat.to(out.device)).to(out.dtype)
        return out
    return hook


def run_exp_c2_on_correct(adapter: BaseAdapter, df_correct: pd.DataFrame,
                           out_file: str, expA_correct: dict = None):
    hidden_dim = adapter.model.lm_head.weight.shape[1]
    d_hat      = _compute_d_hat(adapter)
    r_hat      = _make_ortho_random(hidden_dim, d_hat, seed=SEED)

    print(f"\n[expC2Correct] det_4x4_correct | n={len(df_correct)}")

    results = {}

    for _, row in df_correct.iterrows():
        qid = str(row.get("id", row.get("Problem_ID", "?")))
        # LABEL INVERSION: in correct files, written sign = row["wrong_sign"]
        written_sign     = str(row["wrong_sign"])
        other_sign       = str(row["correct_sign"])
        sign_char_offset = int(row["sign_char_offset"])
        prefix_len       = int(row["prefix_len"])
        full_input       = str(row["full_input"])

        if sign_char_offset <= prefix_len: continue
        if full_input[sign_char_offset] != written_sign: continue

        try:
            wrong_sign_tok, correct_sign_tok = adapter.get_sign_token_ids(
                written_sign, sign_char_offset, full_input)
        except ValueError as e:
            print(f"  ⚠ {qid}: {e}"); continue

        full_ids  = adapter.tokenize(full_input)
        tok_idx   = adapter.char_to_token_idx(sign_char_offset, full_input)
        probe_tok = tok_idx - 1
        if probe_tok < 1: del full_ids; continue

        # Capture magnitudes for CTRL_MAG
        captured = {}
        def make_cap(L, pt):
            def hook(m, i, o):
                if o.dim()==3 and o.shape[1]>pt:
                    captured[L] = o[0, pt, :].detach().float().cpu()
            return hook
        caps = [_mlp(adapter, L).register_forward_hook(make_cap(L, probe_tok))
                for L in [HL1, HL2]]
        try:
            with torch.no_grad():
                out_base = adapter.model(full_ids, use_cache=False)
        finally:
            for h in caps: h.remove()

        logits_base = out_base.logits[0, probe_tok, :].float().cpu()
        base_ld = (logits_base[wrong_sign_tok] - logits_base[correct_sign_tok]).item()
        m75 = abs(torch.dot(captured.get(HL1, torch.zeros(hidden_dim)), d_hat).item()) if HL1 in captured else 0.0
        m78 = abs(torch.dot(captured.get(HL2, torch.zeros(hidden_dim)), d_hat).item()) if HL2 in captured else 0.0
        del out_base

        peak_layer = None
        if expA_correct:
            peak_layer = expA_correct.get(qid, {}).get("peak_layer",
                         expA_correct.get(str(qid), {}).get("peak_layer"))

        q_result = {
            "id": qid, "written_sign": written_sign,
            "baseline_ld": round(base_ld, 4),
            "peak_layer": peak_layer,
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
            broken = base_ld > 0 and ld < 0   # correct run became wrong
            del out, lg
            return round(ld, 4), round(ld - base_ld, 4), broken

        for alpha in ALPHAS:
            # C2: subtract d̂ at L75+L78
            h_c2 = [_mlp(adapter, HL1).register_forward_hook(_make_sub_hook(d_hat, alpha, probe_tok)),
                    _mlp(adapter, HL2).register_forward_hook(_make_sub_hook(d_hat, alpha, probe_tok))]
            ld, delta, broken = run_edit(h_c2)
            q_result["conditions"].setdefault("C2", {})[f"a{alpha}"] = {
                "edit_ld": ld, "delta_ld": delta, "broken": broken}

            # CTRL_MAG
            if m75 > 0 and m78 > 0:
                h_cm = [_mlp(adapter, HL1).register_forward_hook(_make_ctrl_mag_hook(r_hat, m75, alpha, probe_tok)),
                        _mlp(adapter, HL2).register_forward_hook(_make_ctrl_mag_hook(r_hat, m78, alpha, probe_tok))]
                ld_c, delta_c, broken_c = run_edit(h_cm)
                q_result["conditions"].setdefault("CTRL_MAG", {})[f"a{alpha}"] = {
                    "edit_ld": ld_c, "delta_ld": delta_c, "broken": broken_c}

        results[qid] = q_result
        del full_ids, logits_base
        gc.collect(); torch.cuda.empty_cache()

    meta = {
        "n_questions": len(results), "alphas": ALPHAS,
        "d_hat_recipe": "space-prefixed, norm-weighted",
        "label_inversion": "written_sign = row.wrong_sign in correct files",
        "prediction": ("'−'-correct: ld drops (worsened), rarely breaks (bonus). "
                       "'+'-correct: unchanged or slight improvement. CTRL_MAG: null."),
    }
    output = {"meta": meta, "results": results}
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=os.path.dirname(out_file),
                                     suffix=".tmp", delete=False) as f:
        json.dump(output, f, indent=2); tmp = f.name
    shutil.move(tmp, out_file)
    print(f"\n[expC2Correct] {len(results)} questions → {out_file}")
    return results
