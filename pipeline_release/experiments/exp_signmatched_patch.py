"""
exp_signmatched_patch.py — Sign-matched correct-mean patch (Script 2)
=======================================================================
PRE-REGISTERED: earlier F' confound — donor mean pooled 36 '−' / 11 '+' correct
runs but patched mostly '+'-written errors. This fixes it with sign-matched donors.

Method:
  From det_4x4_correct, compute TWO donor means per target layer:
    - mean over '+'-written correct runs (n=11)  [wrong_sign='+' in correct file]
    - mean over '−'-written correct runs (n=36)  [wrong_sign='-' in correct file]
  ⚠ LABEL INVERSION: in correct files, written sign = row["wrong_sign"].
  For each det_4x4_error question, patch with donor mean matching its written sign.

Targets:
  L75.mlp, L78.mlp, L77 self_attn.o_proj, L79.mlp
  Controls: L30.mlp, L50.mlp

PRE-REGISTERED PREDICTIONS:
  If L75/L78 are FIXED '−' pushers → their output is the same on correct runs
  regardless of what the correct answer is → sign-matched patch is near-null.
  L77 stays strongly corrective (sign-matched donor is more specific).
  L30/L50 null.
  Compare: if sign-matched L75/L78 result is smaller than the earlier F', the pooled
  donor confound was real; if similar, L75/L78 error-domain signal is genuine.

Domain: det_4x4_error (patched), det_4x4_correct (donors)
Output: {RESULTS_DIR}/{model}/det_4x4_error/v2_mean_ablation/exp_signmatched_patch.json
"""

import os, gc, json, shutil, tempfile
import torch
import pandas as pd
from adapters.base_adapter import BaseAdapter
from config import RESULTS_DIR

TARGETS = {
    "L75_mlp": {"layer": 75, "type": "mlp"},
    "L78_mlp": {"layer": 78, "type": "mlp"},
    "L77_attn": {"layer": 77, "type": "attn"},
    "L79_mlp": {"layer": 79, "type": "mlp"},
    "L30_mlp": {"layer": 30, "type": "mlp", "control": True},
    "L50_mlp": {"layer": 50, "type": "mlp", "control": True},
}


def _get_module(adapter, layer: int, mtype: str):
    if mtype == "mlp":
        return adapter.model.model.layers[layer].mlp
    else:  # attn
        return adapter.model.model.layers[layer].self_attn.o_proj


def _compute_donor_means(adapter, df_correct: pd.DataFrame) -> dict:
    """
    Pass 1: collect MLP/attn output at probe_tok per sign group from correct set.
    Returns {tname: {'+': mean_tensor, '-': mean_tensor}}
    ⚠ In correct files: written sign = row["wrong_sign"] (label inversion).
    """
    accum = {t: {'+': [], '-': []} for t in TARGETS}

    for _, row in df_correct.iterrows():
        written_sign     = str(row["wrong_sign"])   # label inversion
        sign_char_offset = int(row["sign_char_offset"])
        prefix_len       = int(row["prefix_len"])
        full_input       = str(row["full_input"])

        if sign_char_offset <= prefix_len: continue
        if full_input[sign_char_offset] != written_sign: continue

        full_ids  = adapter.tokenize(full_input)
        tok_idx   = adapter.char_to_token_idx(sign_char_offset, full_input)
        probe_tok = tok_idx - 1
        if probe_tok < 1: del full_ids; continue

        captured = {}
        def make_cap(tname, mod_type, pt):
            def hook(m, i, o):
                out = o[0] if isinstance(o, tuple) else o
                if isinstance(out, torch.Tensor) and out.dim() == 3 and out.shape[1] > pt:
                    captured[tname] = out[0, pt, :].detach().float().cpu()
            return hook

        handles = []
        for tname, tdef in TARGETS.items():
            mod = _get_module(adapter, tdef["layer"], tdef["type"])
            handles.append(mod.register_forward_hook(make_cap(tname, tdef["type"], probe_tok)))

        with torch.no_grad():
            adapter.model.model(full_ids, use_cache=False)
        for h in handles: h.remove()

        for tname in TARGETS:
            if tname in captured:
                accum[tname][written_sign].append(captured[tname])

        del full_ids; gc.collect(); torch.cuda.empty_cache()

    means = {}
    counts = {}  # track n separately so donor_n can be logged correctly
    for tname, by_sign in accum.items():
        means[tname] = {}
        counts[tname] = {}
        for sign, vecs in by_sign.items():
            counts[tname][sign] = len(vecs)
            if vecs:
                means[tname][sign] = torch.stack(vecs).mean(0)
                print(f"  [donor] {tname} sign='{sign}': n={len(vecs)}")
            else:
                print(f"  [donor] {tname} sign='{sign}': NO DATA")
    means["_counts"] = counts  # stash for logging
    return means


def run_exp_signmatched_patch(adapter: BaseAdapter,
                               df_error: pd.DataFrame,
                               df_correct: pd.DataFrame,
                               out_file: str,
                               expA_error: dict):
    late_ids = {qid for qid, q in expA_error.items() if q.get("peak_layer", 0) > 5}
    df_late  = df_error[df_error["id"].astype(str).isin(late_ids)].copy()

    print(f"\n[expSignMatched] Computing sign-matched donor means from {len(df_correct)} correct questions...")
    donor_means = _compute_donor_means(adapter, df_correct)

    results = {}

    for _, row in df_late.iterrows():
        qid              = str(row["id"])
        wrong_sign       = str(row["wrong_sign"])   # in error files this IS the wrong sign
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

        q_result = {
            "id": qid, "written_sign": wrong_sign,
            "baseline_ld": round(base_ld, 4),
            "patches": {}
        }

        for tname, tdef in TARGETS.items():
            if tname == "_counts": continue
            donor_sign = wrong_sign   # match donor to error's written sign
            if tname not in donor_means or donor_sign not in donor_means[tname]:
                q_result["patches"][tname] = {"skip": f"no donor for sign={donor_sign}"}
                continue

            mean_vec = donor_means[tname][donor_sign].to(adapter.model.device)
            donor_n  = donor_means.get("_counts", {}).get(tname, {}).get(donor_sign, "?")

            def make_patch(mv, pt):
                def hook(m, i, o):
                    out = o.clone() if isinstance(o, torch.Tensor) else o[0].clone()
                    if out.dim() == 3 and out.shape[1] > pt:
                        out[0, pt, :] = mv.to(out.dtype)
                    return out
                return hook

            mod = _get_module(adapter, tdef["layer"], tdef["type"])
            handle = mod.register_forward_hook(make_patch(mean_vec, probe_tok))
            with torch.no_grad():
                out_edit = adapter.model(full_ids, use_cache=False)
            handle.remove()

            lg_edit  = out_edit.logits[0, probe_tok, :].float().cpu()
            edit_ld  = (lg_edit[wrong_sign_tok] - lg_edit[correct_sign_tok]).item()
            fixed    = base_ld > 0 and edit_ld < 0

            q_result["patches"][tname] = {
                "prediction": "null" if tdef.get("control") or tname in ("L75_mlp","L78_mlp") else "movement",
                "donor_sign": donor_sign,
                "donor_n":    donor_n,
                "edit_ld":    round(edit_ld, 4),
                "delta_ld":   round(edit_ld - base_ld, 4),
                "fixed":      fixed,
            }
            del out_edit, lg_edit

        results[qid] = q_result
        print(f"  [{qid}] {wrong_sign}  L75Δ={q_result['patches'].get('L75_mlp',{}).get('delta_ld','?')}"
              f"  L77Δ={q_result['patches'].get('L77_attn',{}).get('delta_ld','?')}")
        del full_ids, logits_base
        gc.collect(); torch.cuda.empty_cache()

    meta = {
        "n_questions": len(results),
        "donor_domain": "det_4x4_correct",
        "d_hat_recipe": "N/A — mean-patch, no d_hat",
        "sign_matching": "error written sign matched to correct donor sign",
        "prediction": "L75/L78 near-null; L77 corrective; L30/L50 null",
        "note": "label_inversion: correct files have written_sign = row.wrong_sign",
    }
    output = {"meta": meta, "results": results}
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=os.path.dirname(out_file),
                                     suffix=".tmp", delete=False) as f:
        json.dump(output, f, indent=2); tmp = f.name
    shutil.move(tmp, out_file)
    print(f"\n[expSignMatched] {len(results)} questions → {out_file}")
    return results
