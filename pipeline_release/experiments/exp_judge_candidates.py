"""
exp_judge_candidates.py — Causal test of judge/arbiter candidates (Script 3)
==============================================================================
PRE-REGISTERED: F' showed L77 is the most error-specific component (91% pct_helps)
while L79 was near-null (45%). This tests BOTH as causal judge candidates.

Two arms per candidate (L77.mlp, L79.mlp). Controls: L30.mlp.

ARM A — BREAK correct answers:
  On det_4x4_correct runs, replace candidate output at probe_tok with:
    (A1) SIGN-MATCHED mean over the ERROR set — correct run's written sign
         matched to error-set donor of the same written sign. Avoids the
         donor-composition confound fixed in Script 2.
    (A2) mean ablation over the CORRECT set itself (simpler null)
  Prediction: A1 breaks more answers than A2; L77 breaks more than L79; L30 null.

ARM B — BOOST (scale) on error runs:
  On det_4x4_error runs, scale candidate output at probe_tok by gamma in {1.5, 2.0, 3.0}.
  Prediction: L77 fixes more errors than L79 (scaling up the error-specific component
  helps more than scaling the near-null one).

PRE-REGISTERED ARM B ANALYSIS (run post-pod, downstream of this output):
  Arm B nulls are only interpretable when split by baseline L79/L77 DLA (from expB).
  Prediction: boost fixes errors only where baseline DLA already opposes written sign.
  A null on high-|DLA| questions = this candidate is not the judge.
  A null on low-|DLA| questions = uninformative (gamma * near-zero = near-zero).
  This split is pre-registered here so the analysis cannot be post-hoc motivated.

⚠ LABEL INVERSION in correct files: written sign = row["wrong_sign"].

Domains:
  det_4x4_correct (Arm A), det_4x4_error (Arm B)
Output:
  {RESULTS_DIR}/{model}/det_4x4_correct/v2_mean_ablation/exp_judge_candidates_armA.json
  {RESULTS_DIR}/{model}/det_4x4_error/v2_mean_ablation/exp_judge_candidates_armB.json
"""

import os, gc, json, shutil, tempfile
import torch
import pandas as pd
from adapters.base_adapter import BaseAdapter
from config import RESULTS_DIR

CANDIDATES = {
    "L77_mlp":  {"layer": 77, "type": "mlp"},
    "L77_attn": {"layer": 77, "type": "attn"},   # attn o_proj; earlier F' patched MLP only
    "L79_mlp":  {"layer": 79, "type": "mlp"},
    "L30_mlp":  {"layer": 30, "type": "mlp", "control": True},
}
# L77 identity: the earlier F' run patched L77.mlp (pct_helps=91%). DLA amplifier claim is L77.attn.
# Testing both resolves which component drives the judge/amplifier signal.
BOOST_GAMMAS = [1.5, 2.0, 3.0]


def _get_module(adapter, L, mtype):
    if mtype == "attn":
        return adapter.model.model.layers[L].self_attn.o_proj
    return adapter.model.model.layers[L].mlp

def _get_mlp(adapter, L):
    return adapter.model.model.layers[L].mlp


def _collect_means(adapter, df: pd.DataFrame, by_sign: bool = False) -> dict:
    """Collect mean MLP/attn output at probe_tok.
    If by_sign=True, returns {cname: {'+': tensor, '-': tensor}} for sign-matching.
    If by_sign=False, returns {cname: tensor} (pooled mean).
    Reads row["wrong_sign"] as the probe sign — valid for both error and correct files.
    """
    if by_sign:
        accum = {c: {'+': [], '-': []} for c in CANDIDATES}
    else:
        accum = {c: [] for c in CANDIDATES}
    for _, row in df.iterrows():
        written = str(row["wrong_sign"])   # written sign (or label-inverted written)
        sign_char_offset = int(row["sign_char_offset"])
        prefix_len = int(row["prefix_len"])
        full_input = str(row["full_input"])

        if sign_char_offset <= prefix_len: continue
        if full_input[sign_char_offset] != written: continue

        full_ids  = adapter.tokenize(full_input)
        tok_idx   = adapter.char_to_token_idx(sign_char_offset, full_input)
        probe_tok = tok_idx - 1
        if probe_tok < 1: del full_ids; continue

        captured = {}
        def make_cap(cname, pt):
            def hook(m, i, o):
                if o.dim() == 3 and o.shape[1] > pt:
                    captured[cname] = o[0, pt, :].detach().float().cpu()
            return hook

        handles = [_get_module(adapter, CANDIDATES[c]["layer"], CANDIDATES[c].get("type","mlp"))
                   .register_forward_hook(make_cap(c, probe_tok))
                   for c in CANDIDATES]
        with torch.no_grad():
            adapter.model.model(full_ids, use_cache=False)
        for h in handles: h.remove()
        for c in CANDIDATES:
            if c in captured:
                if by_sign:
                    accum[c][written].append(captured[c])
                else:
                    accum[c].append(captured[c])
        del full_ids; gc.collect(); torch.cuda.empty_cache()

    if by_sign:
        result = {}
        for c, by_s in accum.items():
            result[c] = {}
            for s, vecs in by_s.items():
                if vecs:
                    result[c][s] = torch.stack(vecs).mean(0)
                    print(f"  [means] {c} sign='{s}': n={len(vecs)}")
        return result
    return {c: torch.stack(v).mean(0) if v else None for c, v in accum.items()}


def _run_patch(adapter, full_ids, probe_tok, wrong_sign_tok, correct_sign_tok,
               base_ld, cname, mean_vec, label):
    mv = mean_vec.to(adapter.model.device)
    def hook(m, i, o):
        out = o.clone()
        if out.dim() == 3 and out.shape[1] > probe_tok:
            out[0, probe_tok, :] = mv.to(out.dtype)
        return out
    handle = _get_module(adapter, CANDIDATES[cname]["layer"],
                         CANDIDATES[cname].get("type", "mlp")).register_forward_hook(hook)
    with torch.no_grad():
        out_e = adapter.model(full_ids, use_cache=False)
    handle.remove()
    lg = out_e.logits[0, probe_tok, :].float().cpu()
    ld = (lg[wrong_sign_tok] - lg[correct_sign_tok]).item()
    del out_e, lg
    # "broken" = used in Arm A (correct run flipped to wrong); "fixed" = Arm B
    # Both are logit_diff sign flip: base_ld > 0 and edit_ld < 0.
    # Caller uses the appropriate key for its context.
    sign_flipped = base_ld > 0 and ld < 0
    return {"label": label, "edit_ld": round(ld, 4),
            "delta_ld": round(ld - base_ld, 4), "sign_flipped": sign_flipped}


def run_arm_A(adapter, df_correct, df_error, expA_error, out_file):
    """Break correct answers by patching error-set mean or self-mean."""
    print(f"\n[judgeA] Computing sign-matched error-set means for Arm A...")
    late_ids = {qid for qid, q in expA_error.items() if q.get("peak_layer", 0) > 5}
    df_err_late = df_error[df_error["id"].astype(str).isin(late_ids)].copy()

    # by_sign=True: sign-matches donors to avoid the pooled-mean confound (same fix as Script 2)
    error_means   = _collect_means(adapter, df_err_late, by_sign=True)
    correct_means = _collect_means(adapter, df_correct)

    results = {}
    for _, row in df_correct.iterrows():
        qid          = str(row.get("id", row.get("Problem_ID", "?")))
        written      = str(row["wrong_sign"])   # label inversion
        correct_sign = str(row["correct_sign"])
        sign_char_offset = int(row["sign_char_offset"])
        prefix_len   = int(row["prefix_len"])
        full_input   = str(row["full_input"])

        if sign_char_offset <= prefix_len: continue
        if full_input[sign_char_offset] != written: continue

        try:
            wrong_sign_tok, correct_sign_tok = adapter.get_sign_token_ids(
                written, sign_char_offset, full_input)
        except ValueError: continue

        full_ids  = adapter.tokenize(full_input)
        tok_idx   = adapter.char_to_token_idx(sign_char_offset, full_input)
        probe_tok = tok_idx - 1
        if probe_tok < 1: del full_ids; continue

        with torch.no_grad():
            out_base = adapter.model(full_ids, use_cache=False)
        base_ld = (out_base.logits[0, probe_tok, wrong_sign_tok]
                   - out_base.logits[0, probe_tok, correct_sign_tok]).float().item()
        del out_base

        q = {"id": qid, "written_sign": written, "baseline_ld": round(base_ld, 4), "patches": {}}

        for cname in CANDIDATES:
            q["patches"][cname] = {}
            for label, mean_src, sign_matched in [
                ("A1_error_mean", error_means,   True),   # sign-matched
                ("A2_self_mean",  correct_means, False),  # pooled self-mean
            ]:
                if sign_matched:
                    # use donor matching the correct run's written sign
                    sign_key = written
                    mv = mean_src.get(cname, {}).get(sign_key) if isinstance(mean_src.get(cname), dict) else None
                else:
                    mv = mean_src.get(cname)
                if mv is None: continue
                res = _run_patch(adapter, full_ids, probe_tok,
                                 wrong_sign_tok, correct_sign_tok, base_ld,
                                 cname, mv, label)
                res["broken"] = res.pop("sign_flipped")
                q["patches"][cname][label] = res

        results[qid] = q
        print(f"  [{qid}] {written}  L77@A1_Δ={q['patches'].get('L77_mlp',{}).get('A1_error_mean',{}).get('delta_ld','?')}")
        del full_ids
        gc.collect(); torch.cuda.empty_cache()

    meta = {"arm": "A", "n_questions": len(results),
            "prediction": "L77 A1 breaks more than A2; L79 weak; L30 null"}
    output = {"meta": meta, "results": results}
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=os.path.dirname(out_file),
                                     suffix=".tmp", delete=False) as f:
        json.dump(output, f, indent=2); tmp = f.name
    shutil.move(tmp, out_file)
    print(f"\n[judgeA] {len(results)} questions → {out_file}")
    return results


def run_arm_B(adapter, df_error, expA_error, out_file):
    """Boost (scale) judge candidates on error runs."""
    late_ids = {qid for qid, q in expA_error.items() if q.get("peak_layer", 0) > 5}
    df_late  = df_error[df_error["id"].astype(str).isin(late_ids)].copy()

    print(f"\n[judgeB] Arm B: boost candidates on {len(df_late)} error questions")
    results = {}

    for _, row in df_late.iterrows():
        qid              = str(row["id"])
        wrong_sign       = str(row["wrong_sign"])
        sign_char_offset = int(row["sign_char_offset"])
        prefix_len       = int(row["prefix_len"])
        full_input       = str(row["full_input"])

        if sign_char_offset <= prefix_len: continue
        if full_input[sign_char_offset] != wrong_sign: continue

        try:
            wrong_sign_tok, correct_sign_tok = adapter.get_sign_token_ids(
                wrong_sign, sign_char_offset, full_input)
        except ValueError: continue

        full_ids  = adapter.tokenize(full_input)
        tok_idx   = adapter.char_to_token_idx(sign_char_offset, full_input)
        probe_tok = tok_idx - 1
        if probe_tok < 1: del full_ids; continue

        with torch.no_grad():
            out_base = adapter.model(full_ids, use_cache=False)
        base_ld = (out_base.logits[0, probe_tok, wrong_sign_tok]
                   - out_base.logits[0, probe_tok, correct_sign_tok]).float().item()
        del out_base

        q = {"id": qid, "written_sign": wrong_sign, "baseline_ld": round(base_ld, 4), "boosts": {}}

        for cname, cdef in CANDIDATES.items():
            q["boosts"][cname] = {}
            for gamma in BOOST_GAMMAS:
                def make_scale_hook(g, pt):
                    def hook(m, i, o):
                        out = o.clone()
                        if out.dim() == 3 and out.shape[1] > pt:
                            out[0, pt, :] = out[0, pt, :] * g
                        return out
                    return hook
                handle = _get_module(adapter, cdef["layer"],
                                     cdef.get("type", "mlp")).register_forward_hook(
                    make_scale_hook(gamma, probe_tok))
                with torch.no_grad():
                    out_e = adapter.model(full_ids, use_cache=False)
                handle.remove()
                lg = out_e.logits[0, probe_tok, :].float().cpu()
                ld = (lg[wrong_sign_tok] - lg[correct_sign_tok]).item()
                fixed = base_ld > 0 and ld < 0   # logit_diff flipped sign
                q["boosts"][cname][f"g{gamma}"] = {
                    "edit_ld":    round(ld, 4),
                    "delta_ld":   round(ld - base_ld, 4),
                    "fixed":      fixed}
                del out_e, lg

        results[qid] = q
        print(f"  [{qid}]  L77@γ=2: {q['boosts'].get('L77_mlp',{}).get('g2.0',{}).get('delta_ld','?')}")
        del full_ids
        gc.collect(); torch.cuda.empty_cache()

    meta = {"arm": "B", "n_questions": len(results), "gammas": BOOST_GAMMAS,
            "prediction": "L77 fixes more errors than L79 at γ≥2; L30 null"}
    output = {"meta": meta, "results": results}
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=os.path.dirname(out_file),
                                     suffix=".tmp", delete=False) as f:
        json.dump(output, f, indent=2); tmp = f.name
    shutil.move(tmp, out_file)
    print(f"\n[judgeB] {len(results)} questions → {out_file}")
    return results
