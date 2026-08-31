"""
expG_prime.py — Bias subtraction (causality test)
===================================================

PRE-REGISTERED DESIGN:

  Hypothesis: L75.mlp and L78.mlp push '−' unconditionally and CAUSE '−'-written errors.

  Method: At probe_tok, remove the projection of the MLP output onto the
  '−' direction in unembedding space:

      d̂ = (W_U[minus_tok] − W_U[plus_tok]) / ||...||
      out_new = out − α · (out · d̂) · d̂      α ∈ {0.5, 1.0, 2.0}

  Only the '−'-direction projection is removed; everything perpendicular
  passes through. Direction comes from frozen weights, not activations.

  Targets (run separately, never combined):
    - L75.mlp
    - L78.mlp

  Controls (same code, different layer):
    - L30.mlp (boring layer, should do nothing)
    - Random unit vector at α=1.0 (random direction, should do nothing)

PRE-REGISTERED PREDICTIONS:
  '−'-written errors: logit_diff drops, flip rate increases (dose-dependent in α)
  '+'-written errors: little or no change; a small positive Δlogit_diff (worsening)
                      is NOT falsification — removing a corrective '−' push should
                      mildly worsen '+'-errors. Only a large improvement would falsify.
  IBP '−'-writers:    same direction as det '−'-writers → cross-domain causal claim
  Controls:           null effect

FALSIFICATION: if '+'-written errors improve as much as '−'-written errors →
  L75/L78 do more than a fixed '−' push.

Domains: det_4x4_error, det_5x5_error, ibp_error
Output: {RESULTS_DIR}/{model}/{domain}/v2_mean_ablation/expG_prime_bias_subtraction.json
"""

import os, gc, json, shutil, tempfile
import torch
import pandas as pd
from adapters.base_adapter import BaseAdapter
from config import RESULTS_DIR

ALPHAS   = [0.5, 1.0, 2.0]
TARGETS  = {
    "L75_mlp": {"layer": 75, "type": "mlp", "role": "fixed_minus"},
    "L78_mlp": {"layer": 78, "type": "mlp", "role": "fixed_minus"},
    "L30_mlp": {"layer": 30, "type": "mlp", "role": "control_boring"},
    "L50_mlp": {"layer": 50, "type": "mlp", "role": "control_boring"},
}


def _get_mlp_module(adapter, layer: int):
    return adapter.model.model.layers[layer].mlp


def _compute_d_hat(adapter) -> torch.Tensor:
    """
    Fixed '−' direction matching expB DLA exactly.

    Two requirements (Gemini audit):
    1. Space-prefixed tokens: expB uses get_sign_token_ids which returns the
       space-prefixed sign (' -' tok=481, ' +' tok=488 in Qwen). Bare '-'/'+'
       are different vocab entries with different unembedding vectors.
    2. Final RMSNorm weight: the residual stream passes through model.model.norm
       before W_U — matches adapter.compute_eff_dir exactly.
    """
    tokenizer = adapter.tokenizer
    minus_ids = tokenizer.encode(' -', add_special_tokens=False)
    plus_ids  = tokenizer.encode(' +', add_special_tokens=False)
    if not minus_ids or not plus_ids:
        raise ValueError("Could not tokenize ' -' or ' +'")
    minus_tok = minus_ids[-1]   # last token = the sign character
    plus_tok  = plus_ids[-1]
    W_U    = adapter.model.lm_head.weight.detach().float().cpu()     # [vocab, hidden]
    norm_w = adapter.model.model.norm.weight.detach().float().cpu()  # [hidden]
    d      = (W_U[minus_tok] - W_U[plus_tok]) * norm_w
    return (d / d.norm()).cpu()


def _random_d_hat(hidden_dim: int, seed: int = 42) -> torch.Tensor:
    torch.manual_seed(seed)
    v = torch.randn(hidden_dim)
    return v / v.norm()


def run_expG_prime(adapter: BaseAdapter, df: pd.DataFrame,
                   domain: str, out_file: str, expA_data: dict):

    late_ids = {qid for qid, q in expA_data.items() if q.get("peak_layer", 0) > 5}
    df_late  = df[df["id"].astype(str).isin(late_ids)].copy()

    print(f"\n[expG'] {adapter.model_name} | {domain} | {len(df_late)} late-circuit questions")
    print(f"[expG'] Targets: {list(TARGETS.keys())} | Alphas: {ALPHAS}")

    d_hat = _compute_d_hat(adapter)
    rand_d_hat = _random_d_hat(adapter.cfg["hidden_dim"])

    results = {}

    for _, row in df_late.iterrows():
        qid              = str(row["id"])
        wrong_sign       = str(row["wrong_sign"])
        correct_sign     = str(row["correct_sign"])
        sign_char_offset = int(row["sign_char_offset"])
        prefix_len       = int(row["prefix_len"])
        full_input       = str(row["full_input"])

        print(f"\n  [{qid}] written='{wrong_sign}'")

        if sign_char_offset <= prefix_len:
            print("  CONTAMINATION — skip"); continue
        if full_input[sign_char_offset] != wrong_sign:
            print("  CHAR MISMATCH — skip"); continue

        try:
            wrong_sign_tok, correct_sign_tok = adapter.get_sign_token_ids(
                wrong_sign, sign_char_offset, full_input)
        except ValueError as e:
            print(f"  ⚠ {e} — skip"); continue

        full_ids  = adapter.tokenize(full_input)
        tok_idx   = adapter.char_to_token_idx(sign_char_offset, full_input)
        probe_tok = tok_idx - 1
        if probe_tok < 1:
            print("  ⚠ probe_tok < 1 — skip")
            del full_ids; continue

        # ── Baseline ─────────────────────────────────────────────────────────
        with torch.no_grad():
            out_base = adapter.model(full_ids, use_cache=False)
        logits_base = out_base.logits[0, probe_tok, :].float().cpu()
        base_ld = (logits_base[wrong_sign_tok] - logits_base[correct_sign_tok]).item()
        base_correct = (logits_base[correct_sign_tok] > logits_base[wrong_sign_tok]).item()

        q_result = {
            "id":            qid,
            "written_sign":  wrong_sign,
            "baseline_ld":   round(base_ld, 4),
            "baseline_correct": base_correct,
            "edits": {}
        }

        # ── Subtract bias for each target × alpha ────────────────────────────
        def make_hook(direction, alpha, pt):
            def hook(module, inp, output):
                output = output.clone()
                if output.dim() == 3 and output.shape[1] > pt:
                    h    = output[0, pt, :].float()
                    proj = torch.dot(h, direction.to(h.device)) * direction.to(h.device)
                    output[0, pt, :] -= (alpha * proj).to(output.dtype)
                return output
            return hook

        for tname, tdef in TARGETS.items():
            L   = tdef["layer"]
            mlp = _get_mlp_module(adapter, L)
            q_result["edits"][tname] = {}

            for alpha in ALPHAS:
                handle = mlp.register_forward_hook(make_hook(d_hat, alpha, probe_tok))
                with torch.no_grad():
                    out_edit = adapter.model(full_ids, use_cache=False)
                handle.remove()

                logits_edit = out_edit.logits[0, probe_tok, :].float().cpu()
                edit_ld = (logits_edit[wrong_sign_tok] - logits_edit[correct_sign_tok]).item()
                edit_correct = (logits_edit[correct_sign_tok] > logits_edit[wrong_sign_tok]).item()
                flipped = (not base_correct) and edit_correct

                q_result["edits"][tname][f"alpha_{alpha}"] = {
                    "edit_ld":      round(edit_ld, 4),
                    "delta_ld":     round(edit_ld - base_ld, 4),
                    "edit_correct": edit_correct,
                    "flipped":      flipped,
                }
                print(f"    {tname} α={alpha}: Δld={edit_ld-base_ld:+.3f}  flipped={flipped}")

            del out_edit, logits_edit

        # ── Random direction control (α=1.0 only) ────────────────────────────
        handle = _get_mlp_module(adapter, 75).register_forward_hook(
            make_hook(rand_d_hat, 1.0, probe_tok))
        with torch.no_grad():
            out_rand = adapter.model(full_ids, use_cache=False)
        handle.remove()
        logits_rand = out_rand.logits[0, probe_tok, :].float().cpu()
        rand_ld = (logits_rand[wrong_sign_tok] - logits_rand[correct_sign_tok]).item()
        q_result["random_ctrl_delta_ld"] = round(rand_ld - base_ld, 4)

        results[qid] = q_result
        del full_ids, out_base, logits_base, out_rand, logits_rand
        gc.collect(); torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=os.path.dirname(out_file),
                                     suffix=".tmp", delete=False) as f:
        json.dump(results, f, indent=2); tmp = f.name
    shutil.move(tmp, out_file)
    print(f"\n[expG'] {len(results)} questions → {out_file}")
    return results
