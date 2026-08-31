"""
exp_combined_ablation.py — Stacked intervention (Script 1)
============================================================
PRE-REGISTERED: single-component G' never flipped (max Δld=−2.55 vs baseline
margin ~+7). This stacks components to attempt flips.

Conditions (each run separately per question):
  C1 — subtract d̂ from L75.mlp at probe_tok            (replicates G', sanity)
  C2 — subtract d̂ from L75.mlp AND L78.mlp
  C3 — C2 + scale L77/L79 self_attn.o_proj output at probe_tok by 0.5
  C4 — C3 with amplifier scale 0.0 (zero out L77/L79 attn output at probe_tok)
  CTRL_RAND — C2 with random unit vector instead of d̂ (seed=0)
  CTRL_BORING — C2 at L30+L50 instead of L75+L78

Alphas for MLP subtraction: {1.0, 2.0, 3.0}

PRE-REGISTERED PREDICTIONS:
  C3/C4 at alpha=1 flip >=1 question — alpha=1 removes the '-' push only.
  alpha>1 injects a '+' push (not just removing '-'); if flips only appear at
  alpha>1 the honest claim is "reversing the habit flips," which is weaker.
  Controls (CTRL_RAND, CTRL_BORING, CTRL_ATTN) flip none.
  Dose-response: flip_rate(alpha=3) >= flip_rate(alpha=2) >= flip_rate(alpha=1).
  CTRL_ATTN zeroes attention at non-story layers (L60/L65) + random MLP subtraction
  — controls the attention-zeroing component of C3/C4 separately.

FAILURE BRANCH (pre-registered): if C4/alpha=3 flips nothing, report
  "fraction of baseline margin closed" = mean(-delta_ld)/mean(baseline_ld).
  Claim becomes: distributed error, no privileged components.

For any flipped question, run one hooked generate() to confirm the model
actually emits the correct sign in free generation (see post-hoc analysis).

Domains: det_4x4_error, det_5x5_error, ibp_error
Output: {RESULTS_DIR}/{model}/{domain}/v2_mean_ablation/exp_combined_ablation.json
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
AMP_SCALES = {"C3": 0.5, "C4": 0.0}


def _compute_d_hat(adapter) -> torch.Tensor:
    tok = adapter.tokenizer
    minus_ids = tok.encode(' -', add_special_tokens=False)
    plus_ids  = tok.encode(' +', add_special_tokens=False)
    W_U    = adapter.model.lm_head.weight.detach().float().cpu()
    norm_w = adapter.model.model.norm.weight.detach().float().cpu()
    d = (W_U[minus_ids[-1]] - W_U[plus_ids[-1]]) * norm_w
    return (d / d.norm()).cpu()


def _random_unit(hidden_dim: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    v = torch.randn(hidden_dim)
    return v / v.norm()


def _mlp(adapter, L):
    return adapter.model.model.layers[L].mlp


def _o_proj(adapter, L):
    return adapter.model.model.layers[L].self_attn.o_proj


def _make_sub_hook(direction, alpha, pt):
    """Subtract α*(out·d̂)*d̂ from MLP output at probe_tok."""
    def hook(module, inp, output):
        output = output.clone()
        if output.dim() == 3 and output.shape[1] > pt:
            h = output[0, pt, :].float()
            proj = torch.dot(h, direction.to(h.device))
            output[0, pt, :] -= (alpha * proj * direction.to(h.device)).to(output.dtype)
        return output
    return hook


def _make_scale_hook(scale, pt):
    """Scale self_attn.o_proj output at probe_tok by `scale`."""
    def hook(module, inp, output):
        output = output.clone()
        if output.dim() == 3 and output.shape[1] > pt:
            output[0, pt, :] = output[0, pt, :] * scale
        return output
    return hook


def run_exp_combined_ablation(adapter: BaseAdapter, df: pd.DataFrame,
                               domain: str, out_file: str, expA_data: dict):
    late_ids = {qid for qid, q in expA_data.items() if q.get("peak_layer", 0) > 5}
    df_late  = df[df["id"].astype(str).isin(late_ids)].copy()

    hidden_dim = adapter.model.lm_head.weight.shape[1]
    d_hat      = _compute_d_hat(adapter)
    rand_dir   = _random_unit(hidden_dim, seed=0)

    print(f"\n[expCombined] {domain} | n={len(df_late)} late-circuit questions")

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

        with torch.no_grad():
            out_base = adapter.model(full_ids, use_cache=False)
        logits_base = out_base.logits[0, probe_tok, :].float().cpu()
        base_ld     = (logits_base[wrong_sign_tok] - logits_base[correct_sign_tok]).item()
        del out_base

        base_top1_is_wrong = (int(logits_base.argmax().item()) == wrong_sign_tok)
        q_result = {
            "id": qid, "written_sign": wrong_sign,
            "baseline_ld": round(base_ld, 4),
            "baseline_top1_is_wrong_sign": base_top1_is_wrong,
            "conditions": {}
        }

        # Helper: run with given handles
        def run_edit(handles):
            try:
                with torch.no_grad():
                    out = adapter.model(full_ids, use_cache=False)
            finally:
                for h in handles: h.remove()
            lg = out.logits[0, probe_tok, :].float().cpu()
            ld = (lg[wrong_sign_tok] - lg[correct_sign_tok]).item()
            flipped = base_ld > 0 and ld < 0
            # Argmax check: did the top-1 token actually change to correct_sign?
            top1 = int(lg.argmax().item())
            top1_now_correct = (top1 == correct_sign_tok)
            del out, lg
            return round(ld, 4), round(ld - base_ld, 4), flipped, top1_now_correct

        def record(ld, delta, flipped, top1_correct):
            return {"edit_ld": ld, "delta_ld": delta,
                    "flipped": flipped, "top1_now_correct": top1_correct}

        for alpha in ALPHAS:
            # C1
            h1 = [_mlp(adapter, HL1).register_forward_hook(_make_sub_hook(d_hat, alpha, probe_tok))]
            q_result["conditions"].setdefault("C1", {})[f"a{alpha}"] = record(*run_edit(h1))

            # C2
            h2 = [
                _mlp(adapter, HL1).register_forward_hook(_make_sub_hook(d_hat, alpha, probe_tok)),
                _mlp(adapter, HL2).register_forward_hook(_make_sub_hook(d_hat, alpha, probe_tok)),
            ]
            q_result["conditions"].setdefault("C2", {})[f"a{alpha}"] = record(*run_edit(h2))

            # C3 and C4
            for cname, amp_scale in AMP_SCALES.items():
                h3 = [
                    _mlp(adapter, HL1).register_forward_hook(_make_sub_hook(d_hat, alpha, probe_tok)),
                    _mlp(adapter, HL2).register_forward_hook(_make_sub_hook(d_hat, alpha, probe_tok)),
                    _o_proj(adapter, 77).register_forward_hook(_make_scale_hook(amp_scale, probe_tok)),
                    _o_proj(adapter, 79).register_forward_hook(_make_scale_hook(amp_scale, probe_tok)),
                ]
                q_result["conditions"].setdefault(cname, {})[f"a{alpha}"] = record(*run_edit(h3))

        # Controls (alpha=2.0 only)
        h_rand = [
            _mlp(adapter, HL1).register_forward_hook(_make_sub_hook(rand_dir, 2.0, probe_tok)),
            _mlp(adapter, HL2).register_forward_hook(_make_sub_hook(rand_dir, 2.0, probe_tok)),
        ]
        q_result["conditions"]["CTRL_RAND"] = record(*run_edit(h_rand))

        h_boring = [
            _mlp(adapter, CTRL1).register_forward_hook(_make_sub_hook(d_hat, 2.0, probe_tok)),
            _mlp(adapter, CTRL2).register_forward_hook(_make_sub_hook(d_hat, 2.0, probe_tok)),
        ]
        q_result["conditions"]["CTRL_BORING"] = record(*run_edit(h_boring))

        # CTRL_ATTN: random MLP subtraction + zero attn output at non-story layers L60/L65
        # Controls the attention-zeroing component of C3/C4 independently.
        h_ctrl_attn = [
            _mlp(adapter, HL1).register_forward_hook(_make_sub_hook(rand_dir, 2.0, probe_tok)),
            _mlp(adapter, HL2).register_forward_hook(_make_sub_hook(rand_dir, 2.0, probe_tok)),
            _o_proj(adapter, 60).register_forward_hook(_make_scale_hook(0.0, probe_tok)),
            _o_proj(adapter, 65).register_forward_hook(_make_scale_hook(0.0, probe_tok)),
        ]
        q_result["conditions"]["CTRL_ATTN"] = record(*run_edit(h_ctrl_attn))

        print(f"  [{qid}] base={base_ld:+.3f}  C4@α=2: {q_result['conditions'].get('C4',{}).get('a2.0',{}).get('delta_ld','?')}")
        results[qid] = q_result
        del full_ids, logits_base
        gc.collect(); torch.cuda.empty_cache()

    meta = {
        "n_questions": len(results),
        "alpha_values": ALPHAS, "amp_scales": AMP_SCALES,
        "d_hat_recipe": "space-prefixed, norm-weighted",
        "prediction": ("HEADLINE: C3/C4 at alpha=1 flip >=1 question (alpha=1 = removal only). "
                       "alpha>1 injects opposite push — dose-response but not headline. "
                       "Controls (CTRL_RAND, CTRL_BORING, CTRL_ATTN) flip none. "
                       "FAILURE BRANCH: if C4/alpha=3 flips nothing, report fraction_of_margin_closed."),
    }
    output = {"meta": meta, "results": results}
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=os.path.dirname(out_file),
                                     suffix=".tmp", delete=False) as f:
        json.dump(output, f, indent=2); tmp = f.name
    shutil.move(tmp, out_file)
    print(f"\n[expCombined] {len(results)} questions → {out_file}")
    return results
