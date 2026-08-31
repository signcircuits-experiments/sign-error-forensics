"""
exp_judge_crossdomain.py — Arm B cross-domain (S4)
===================================================
Arm B (boost L79.mlp, gammas {1.5, 2.0, 3.0}) on det_5x5_error and ibp_error.
L77 dropped (resolved in an earlier run: L77.attn near-null, L77.mlp no fixes).
L30 control included.

Identical to the earlier run_arm_B but only {L79_mlp, L30_mlp}.

PRE-REGISTERED: fixes appear at gamma=3 in 5x5; IBP marked exploratory (n=10).
DLA-split analysis (per pre-registration in exp_judge_candidates.py) applies
post-hoc, downstream of this output.

Output: {RESULTS_DIR}/{model}/{domain}/v2_mean_ablation/exp_judge_crossdomain.json
"""

import os, gc, json, shutil, tempfile
import torch
import pandas as pd
from adapters.base_adapter import BaseAdapter
from config import RESULTS_DIR

CANDIDATES = {
    "L79_mlp": {"layer": 79},
    "L30_mlp": {"layer": 30, "control": True},
}
BOOST_GAMMAS = [1.5, 2.0, 3.0]


def _mlp(adapter, L):
    return adapter.model.model.layers[L].mlp


def run_exp_judge_crossdomain(adapter: BaseAdapter, df: pd.DataFrame,
                               domain: str, out_file: str, expA_data: dict):
    late_ids = {qid for qid, q in expA_data.items() if q.get("peak_layer", 0) > 5}
    df_late  = df[df["id"].astype(str).isin(late_ids)].copy()
    exploratory = len(df_late) < 15

    print(f"\n[judgeXD] {domain} | n={len(df_late)}"
          + (" (EXPLORATORY)" if exploratory else ""))

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
        except ValueError as e:
            print(f"  ⚠ {qid}: {e}"); continue

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
            L = cdef["layer"]
            for gamma in BOOST_GAMMAS:
                def make_scale_hook(g, pt):
                    def hook(m, i, o):
                        out = o.clone()
                        if out.dim() == 3 and out.shape[1] > pt:
                            out[0, pt, :] = out[0, pt, :] * g
                        return out
                    return hook
                handle = _mlp(adapter, L).register_forward_hook(make_scale_hook(gamma, probe_tok))
                try:
                    with torch.no_grad():
                        out_e = adapter.model(full_ids, use_cache=False)
                finally:
                    handle.remove()
                lg  = out_e.logits[0, probe_tok, :].float().cpu()
                ld  = (lg[wrong_sign_tok] - lg[correct_sign_tok]).item()
                fixed = base_ld > 0 and ld < 0
                q["boosts"][cname][f"g{gamma}"] = {
                    "edit_ld": round(ld, 4), "delta_ld": round(ld - base_ld, 4), "fixed": fixed}
                del out_e, lg

        results[qid] = q
        del full_ids
        gc.collect(); torch.cuda.empty_cache()

    meta = {
        "n_questions": len(results), "gammas": BOOST_GAMMAS,
        "domain": domain, "exploratory": exploratory,
        "candidates_tested": list(CANDIDATES.keys()),
        "l77_dropped": "resolved in an earlier run: L77.attn near-null, L77.mlp no fixes",
        "prediction": ("fixes at gamma=3 in 5x5; IBP exploratory (n=10). "
                       "DLA-split: fixes concentrate where L79 DLA already opposes written sign."),
    }
    output = {"meta": meta, "results": results}
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=os.path.dirname(out_file),
                                     suffix=".tmp", delete=False) as f:
        json.dump(output, f, indent=2); tmp = f.name
    shutil.move(tmp, out_file)
    print(f"\n[judgeXD] {len(results)} questions → {out_file}")
    return results
