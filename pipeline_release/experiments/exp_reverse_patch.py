"""
exp_reverse_patch.py — Reverse patch: break correct answers (4×4 only)
=======================================================================

Mirror of expF'. Instead of patching correct-mean into error runs,
we patch ERROR-mean into correct runs.

PRE-REGISTERED DESIGN:

  Method: Compute mean L75.mlp and L78.mlp output at probe_tok over the
  det_4x4_error set. Patch each correct run with this "error mean",
  one component at a time.

PRE-REGISTERED PREDICTIONS:
  Correct '+'-answer cases  → logit_diff shifts toward '−' (L75/L78 push '−'),
                               some may break (flip to '−' output).
  Correct '−'-answer cases  → unaffected (the error-mean still pushes '−',
                               same direction they already favor).
  L30/L50 controls          → null.

This is BIDIRECTIONAL causality: G' shows the push FIXES '−'-errors;
reverse patch shows the SAME push BREAKS '+'-correct answers.

Domain: det_4x4_error (source of error means) + det_4x4_correct (patched)
Output: {RESULTS_DIR}/{model}/det_4x4_correct/v2_mean_ablation/exp_reverse_patch.json
"""

import os, gc, json, shutil, tempfile
import torch
import pandas as pd
from adapters.base_adapter import BaseAdapter
from config import RESULTS_DIR

TARGETS = {
    "L75_mlp": {"layer": 75, "role": "fixed_minus"},
    "L78_mlp": {"layer": 78, "role": "fixed_minus"},
    "L30_mlp": {"layer": 30, "role": "control"},
}


def _get_mlp(adapter, L: int):
    return adapter.model.model.layers[L].mlp


def _compute_error_means(adapter, df_error: pd.DataFrame,
                          expA_error: dict, layers: list) -> dict:
    """
    Compute mean MLP output at probe_tok over late-circuit error questions.
    """
    late_ids = {qid for qid, q in expA_error.items() if q.get("peak_layer", 0) > 5}
    accum    = {L: [] for L in layers}

    for _, row in df_error.iterrows():
        if str(row["id"]) not in late_ids: continue
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
                if o.dim()==3 and o.shape[1]>pt:
                    captured[L] = o[0, pt, :].detach().float().cpu()
            return hook

        handles = [_get_mlp(adapter, L).register_forward_hook(make_cap(L, probe_tok))
                   for L in layers]
        with torch.no_grad():
            adapter.model.model(full_ids, use_cache=False)
        for h in handles: h.remove()
        for L in layers:
            if L in captured: accum[L].append(captured[L])
        del full_ids; gc.collect(); torch.cuda.empty_cache()

    means = {}
    for L, vecs in accum.items():
        if vecs:
            means[L] = torch.stack(vecs).mean(0)
            print(f"  [error mean] L{L}: from {len(vecs)} error questions")
    return means


def run_exp_reverse_patch(adapter: BaseAdapter,
                           df_correct: pd.DataFrame,
                           df_error: pd.DataFrame,
                           out_file: str,
                           expA_error: dict):

    layers = [v["layer"] for v in TARGETS.values()]

    print(f"\n[reverse] {adapter.model_name} | 4x4 reverse patch")
    print(f"[reverse] Correct set: {len(df_correct)} | Error set: {len(df_error)}")
    print("[reverse] Computing error-domain means...")
    error_means = _compute_error_means(adapter, df_error, expA_error, layers)

    results = {}

    for _, row in df_correct.iterrows():
        qid              = str(row.get("id", row.get("Problem_ID", "?")))
        wrong_sign       = str(row["wrong_sign"])   # written sign (correct answer)
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
        logits_base  = out_base.logits[0, probe_tok, :].float().cpu()
        base_ld      = (logits_base[wrong_sign_tok] - logits_base[correct_sign_tok]).item()
        base_correct = (logits_base[correct_sign_tok] > logits_base[wrong_sign_tok]).item()

        q_result = {
            "id":               qid,
            "written_sign":     wrong_sign,   # the correct mathematical answer
            "baseline_ld":      round(base_ld, 4),
            "baseline_correct": base_correct,
            "patches":          {}
        }

        for tname, tdef in TARGETS.items():
            L = tdef["layer"]
            if L not in error_means:
                q_result["patches"][tname] = {"skip": "no error mean"}
                continue

            mean_vec = error_means[L].to(adapter.model.device)

            def make_patch(mean, pt):
                def hook(module, inp, output):
                    output = output.clone()
                    if output.dim()==3 and output.shape[1]>pt:
                        output[0, pt, :] = mean.to(output.dtype)
                    return output
                return hook

            handle = _get_mlp(adapter, L).register_forward_hook(make_patch(mean_vec, probe_tok))
            with torch.no_grad():
                out_edit = adapter.model(full_ids, use_cache=False)
            handle.remove()

            logits_edit  = out_edit.logits[0, probe_tok, :].float().cpu()
            edit_ld      = (logits_edit[wrong_sign_tok] - logits_edit[correct_sign_tok]).item()
            edit_correct = (logits_edit[correct_sign_tok] > logits_edit[wrong_sign_tok]).item()
            broken       = base_correct and (not edit_correct)   # was correct, now wrong

            q_result["patches"][tname] = {
                "edit_ld":    round(edit_ld, 4),
                "delta_ld":   round(edit_ld - base_ld, 4),
                "edit_correct": edit_correct,
                "broken":     broken,
            }
            del out_edit, logits_edit

        results[qid] = q_result
        print(f"  [{qid}] written='{wrong_sign}'  L75Δ={q_result['patches'].get('L75_mlp',{}).get('delta_ld','?')}")
        del full_ids, out_base, logits_base
        gc.collect(); torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=os.path.dirname(out_file),
                                     suffix=".tmp", delete=False) as f:
        json.dump(results, f, indent=2); tmp = f.name
    shutil.move(tmp, out_file)
    print(f"\n[reverse] {len(results)} questions → {out_file}")
    return results
