"""
expF_prime.py — Correct-mean patch (fixedness null test)
=========================================================

PRE-REGISTERED DESIGN:

  Hypothesis: L75 and L78 MLPs are "fixed" — their output at probe_tok is
  the same on correct and error runs. Swapping in the correct-domain mean
  changes nothing.

  Method: For each target component, compute the MEAN OUTPUT at probe_tok
  over all questions in the correct-domain dataset. Then on each error
  question, replace that component's output at probe_tok with this mean.
  One component at a time, never combined.

  Targets:
    L75.mlp   — fixed '−' predictor (prediction: NULL effect)
    L78.mlp   — fixed '−' predictor (prediction: NULL effect)
    L79.mlp   — arbiter            (prediction: REAL movement, some flips)
    L77.mlp   — written-sign amp   (prediction: some movement)
    L30.mlp   — boring control     (prediction: null)
    L50.mlp   — boring control     (prediction: null)

PRE-REGISTERED PREDICTIONS:
  L75, L78 → Δlogit_diff ≈ 0 (null). The null IS the finding — it shows
             their output carries no error-specific signal.
  L79      → real movement; some errors flip to correct.
  L30/L50  → null (sanity controls).

SANITY CHECK (run first):
  Compare mean MLP output (projected onto d̂) between correct and error sets
  for L75 and L78. If already near-identical → null prediction is pre-confirmed
  at the activation level before running the patch.

Domains: det_4x4_error (paired with det_4x4_correct)
         ibp_error     (paired with ibp_correct)
Output: {RESULTS_DIR}/{model}/{domain}/v2_mean_ablation/expF_prime_correct_mean_patch.json
"""

import os, gc, json, shutil, tempfile
import torch
import numpy as np
import pandas as pd
from adapters.base_adapter import BaseAdapter
from config import RESULTS_DIR, DATA_FILES

TARGETS = {
    "L75_mlp": {"layer": 75, "prediction": "null",     "role": "fixed_minus"},
    "L78_mlp": {"layer": 78, "prediction": "null",     "role": "fixed_minus"},
    "L79_mlp": {"layer": 79, "prediction": "movement", "role": "arbiter"},
    "L77_mlp": {"layer": 77, "prediction": "movement", "role": "amplifier"},
    "L30_mlp": {"layer": 30, "prediction": "null",     "role": "control"},
    "L50_mlp": {"layer": 50, "prediction": "null",     "role": "control"},
}


def _get_mlp(adapter, layer: int):
    return adapter.model.model.layers[layer].mlp


def _compute_d_hat(adapter) -> torch.Tensor:
    """Space-prefixed tokens + RMSNorm weight — matches expB DLA direction."""
    tokenizer = adapter.tokenizer
    minus_ids = tokenizer.encode(' -', add_special_tokens=False)
    plus_ids  = tokenizer.encode(' +', add_special_tokens=False)
    W_U    = adapter.model.lm_head.weight.detach().float().cpu()
    norm_w = adapter.model.model.norm.weight.detach().float().cpu()
    d = (W_U[minus_ids[-1]] - W_U[plus_ids[-1]]) * norm_w
    return (d / d.norm()).cpu()


def _compute_correct_means(adapter, df_correct: pd.DataFrame,
                            layers: list) -> dict:
    """
    Pass 1: collect MLP outputs at probe_tok over the correct set.
    Returns dict: layer → mean tensor [hidden_dim].
    """
    accum = {L: [] for L in layers}

    for _, row in df_correct.iterrows():
        sign_char_offset = int(row["sign_char_offset"])
        prefix_len       = int(row["prefix_len"])
        full_input       = str(row["full_input"])
        wrong_sign       = str(row["wrong_sign"])   # written sign in correct files

        if sign_char_offset <= prefix_len: continue
        if full_input[sign_char_offset] != wrong_sign: continue

        full_ids  = adapter.tokenize(full_input)
        tok_idx   = adapter.char_to_token_idx(sign_char_offset, full_input)
        probe_tok = tok_idx - 1
        if probe_tok < 1:
            del full_ids; continue

        captured = {}
        def make_capture_hook(L, pt):
            def hook(module, inp, output):
                if output.dim() == 3 and output.shape[1] > pt:
                    captured[L] = output[0, pt, :].detach().float().cpu()
            return hook

        handles = [_get_mlp(adapter, L).register_forward_hook(make_capture_hook(L, probe_tok))
                   for L in layers]
        with torch.no_grad():
            adapter.model.model(full_ids, use_cache=False)
        for h in handles: h.remove()

        for L in layers:
            if L in captured:
                accum[L].append(captured[L])

        del full_ids; gc.collect(); torch.cuda.empty_cache()

    means = {}
    for L, vecs in accum.items():
        if vecs:
            means[L] = torch.stack(vecs).mean(0)   # [hidden_dim]
            print(f"  [mean] L{L}: computed from {len(vecs)} correct questions")
        else:
            print(f"  [mean] L{L}: NO data — skipping")
    return means


def run_expF_prime(adapter: BaseAdapter,
                   df_error: pd.DataFrame,
                   df_correct: pd.DataFrame,
                   domain: str,
                   out_file: str,
                   expA_data_error: dict):

    late_ids = {qid for qid, q in expA_data_error.items()
                if q.get("peak_layer", 0) > 5}
    df_late  = df_error[df_error["id"].astype(str).isin(late_ids)].copy()

    layers = [v["layer"] for v in TARGETS.values()]

    print(f"\n[expF'] {adapter.model_name} | {domain}")
    print(f"[expF'] Error questions: {len(df_late)} | Correct set: {len(df_correct)}")
    print(f"[expF'] Computing correct-domain means for layers {layers}...")

    correct_means = _compute_correct_means(adapter, df_correct, layers)
    d_hat = _compute_d_hat(adapter)

    # ── Sanity check: compare activation means on correct vs error at L75/L78 ─
    print("\n[expF'] Sanity check — correct-mean vs error-mean projection onto d_hat:")
    err_accum = {L: [] for L in [75, 78]}
    for _, row in df_late.iterrows():
        sign_char_offset = int(row["sign_char_offset"])
        prefix_len       = int(row["prefix_len"])
        full_input       = str(row["full_input"])
        wrong_sign       = str(row["wrong_sign"])
        if sign_char_offset <= prefix_len: continue
        if full_input[sign_char_offset] != wrong_sign: continue
        full_ids  = adapter.tokenize(full_input)
        tok_idx   = adapter.char_to_token_idx(sign_char_offset, full_input)
        probe_tok = tok_idx - 1
        if probe_tok < 1: del full_ids; continue
        captured = {}
        def make_cap(L, pt):
            def hook(m, i, o):
                if o.dim()==3 and o.shape[1]>pt: captured[L]=o[0,pt,:].detach().float().cpu()
            return hook
        handles = [_get_mlp(adapter, L).register_forward_hook(make_cap(L, probe_tok))
                   for L in err_accum]
        with torch.no_grad(): adapter.model.model(full_ids, use_cache=False)
        for h in handles: h.remove()
        for L in err_accum:
            if L in captured: err_accum[L].append(captured[L])
        del full_ids; gc.collect(); torch.cuda.empty_cache()

    for L in [75, 78]:
        if err_accum[L] and L in correct_means:
            err_mean = torch.stack(err_accum[L]).mean(0)
            corr_mean = correct_means[L]
            err_proj  = torch.dot(err_mean, d_hat).item()
            corr_proj = torch.dot(corr_mean, d_hat).item()
            diff_proj = abs(err_proj - corr_proj)
            print(f"  L{L}: error_mean·d̂={err_proj:+.4f}  correct_mean·d̂={corr_proj:+.4f}  |diff|={diff_proj:.4f}")
            if diff_proj < 0.05:
                print(f"  L{L}: ✓ near-identical — null prediction pre-confirmed")
            else:
                print(f"  L{L}: ⚠ difference detected — null may not hold")

    # ── Pass 2: patch each error question ────────────────────────────────────
    results = {}

    for _, row in df_late.iterrows():
        qid              = str(row["id"])
        wrong_sign       = str(row["wrong_sign"])
        correct_sign     = str(row["correct_sign"])
        sign_char_offset = int(row["sign_char_offset"])
        prefix_len       = int(row["prefix_len"])
        full_input       = str(row["full_input"])

        print(f"\n  [{qid}] written='{wrong_sign}'")
        if sign_char_offset <= prefix_len: print("  CONTAMINATION — skip"); continue
        if full_input[sign_char_offset] != wrong_sign: print("  CHAR MISMATCH — skip"); continue

        try:
            wrong_sign_tok, correct_sign_tok = adapter.get_sign_token_ids(
                wrong_sign, sign_char_offset, full_input)
        except ValueError as e:
            print(f"  ⚠ {e} — skip"); continue

        full_ids  = adapter.tokenize(full_input)
        tok_idx   = adapter.char_to_token_idx(sign_char_offset, full_input)
        probe_tok = tok_idx - 1
        if probe_tok < 1: print("  ⚠ probe_tok < 1"); del full_ids; continue

        with torch.no_grad():
            out_base = adapter.model(full_ids, use_cache=False)
        logits_base = out_base.logits[0, probe_tok, :].float().cpu()
        base_ld = (logits_base[wrong_sign_tok] - logits_base[correct_sign_tok]).item()
        base_correct = (logits_base[correct_sign_tok] > logits_base[wrong_sign_tok]).item()

        q_result = {
            "id":               qid,
            "written_sign":     wrong_sign,
            "baseline_ld":      round(base_ld, 4),
            "baseline_correct": base_correct,
            "patches":          {}
        }

        for tname, tdef in TARGETS.items():
            L = tdef["layer"]
            if L not in correct_means:
                q_result["patches"][tname] = {"skip": "no correct mean"}
                continue

            mean_vec = correct_means[L].to(adapter.model.device)

            def make_patch_hook(mean, pt):
                def hook(module, inp, output):
                    output = output.clone()
                    if output.dim() == 3 and output.shape[1] > pt:
                        output[0, pt, :] = mean.to(output.dtype)
                    return output
                return hook

            handle = _get_mlp(adapter, L).register_forward_hook(
                make_patch_hook(mean_vec, probe_tok))
            with torch.no_grad():
                out_edit = adapter.model(full_ids, use_cache=False)
            handle.remove()

            logits_edit = out_edit.logits[0, probe_tok, :].float().cpu()
            edit_ld     = (logits_edit[wrong_sign_tok] - logits_edit[correct_sign_tok]).item()
            edit_correct = (logits_edit[correct_sign_tok] > logits_edit[wrong_sign_tok]).item()
            flipped      = (not base_correct) and edit_correct

            q_result["patches"][tname] = {
                "prediction": tdef["prediction"],
                "edit_ld":    round(edit_ld, 4),
                "delta_ld":   round(edit_ld - base_ld, 4),
                "flipped":    flipped,
            }
            print(f"    {tname}: Δld={edit_ld-base_ld:+.3f}  flipped={flipped}  (pred={tdef['prediction']})")
            del out_edit, logits_edit

        results[qid] = q_result
        del full_ids, out_base, logits_base
        gc.collect(); torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=os.path.dirname(out_file),
                                     suffix=".tmp", delete=False) as f:
        json.dump(results, f, indent=2); tmp = f.name
    shutil.move(tmp, out_file)
    print(f"\n[expF'] {len(results)} questions → {out_file}")
    return results
