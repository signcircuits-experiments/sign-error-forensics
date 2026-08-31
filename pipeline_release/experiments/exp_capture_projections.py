"""
exp_capture_projections.py — Volume-knob capture (S5)
======================================================
One clean forward per question across ALL domains.
Captures L75/L78/L79 mlp output at probe_tok. Stores:
  proj_d = h_L · d̂   (signed projection onto '-' direction)
  norm_h = ||h_L||
  baseline_ld, written_sign, domain

NO intervention — pure measurement for the "L78 error-modulated magnitude" claim.
Compact output (floats only, no vectors) — one JSON at
  {RESULTS_DIR}/{model}/exp_capture_projections.json

Domains: det_4x4_error, det_4x4_correct, det_5x5_error, ibp_error, ibp_correct,
         arith_simple, det_3x3_correct
"""

import os, gc, json, shutil, tempfile
import torch
import pandas as pd
from adapters.base_adapter import BaseAdapter
from config import RESULTS_DIR, DATA_FILES, CORRECT_DATASETS

CAPTURE_LAYERS = [75, 78, 79]

DOMAINS = [
    "det_4x4_error",
    "det_4x4_correct",
    "det_5x5_error",
    "ibp_error",
    "ibp_correct",
    "arith_simple",
    "det_3x3_correct",
]


def _compute_d_hat(adapter) -> torch.Tensor:
    tok   = adapter.tokenizer
    m_ids = tok.encode(' -', add_special_tokens=False)
    p_ids = tok.encode(' +', add_special_tokens=False)
    W_U   = adapter.model.lm_head.weight.detach().float().cpu()
    norm_w = adapter.model.model.norm.weight.detach().float().cpu()
    d = (W_U[m_ids[-1]] - W_U[p_ids[-1]]) * norm_w
    return (d / d.norm()).cpu()


def _mlp(adapter, L):
    return adapter.model.model.layers[L].mlp


def _load_df(model_name, domain):
    from config import DATA_FILES
    path = DATA_FILES[model_name].get(domain)
    if not path or not os.path.exists(path):
        return None
    df = pd.read_excel(path) if path.endswith(".xlsx") else pd.read_csv(path)
    if "id" not in df.columns and "Problem_ID" in df.columns:
        df = df.rename(columns={"Problem_ID": "id"})
    return df


def run_exp_capture_projections(adapter: BaseAdapter, model_name: str, out_file: str):
    d_hat = _compute_d_hat(adapter)

    print(f"\n[expCapture] {model_name} | Domains: {DOMAINS}")
    results = []

    for domain in DOMAINS:
        df = _load_df(model_name, domain)
        if df is None:
            print(f"  SKIP {domain}: data file not found")
            continue
        print(f"  {domain}: {len(df)} questions")

        for _, row in df.iterrows():
            qid              = str(row.get("id", "?"))
            written_sign     = str(row.get("wrong_sign", "?"))
            sign_char_offset = int(row.get("sign_char_offset", 0))
            prefix_len       = int(row.get("prefix_len", 0))
            full_input       = str(row.get("full_input", ""))

            if sign_char_offset <= prefix_len: continue
            if not full_input or sign_char_offset >= len(full_input): continue
            if full_input[sign_char_offset] != written_sign: continue

            try:
                wrong_sign_tok, correct_sign_tok = adapter.get_sign_token_ids(
                    written_sign, sign_char_offset, full_input)
            except ValueError:
                continue

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

            handles = [_mlp(adapter, L).register_forward_hook(make_cap(L, probe_tok))
                       for L in CAPTURE_LAYERS]
            try:
                with torch.no_grad():
                    out_base = adapter.model(full_ids, use_cache=False)
            finally:
                for h in handles: h.remove()

            logits_base = out_base.logits[0, probe_tok, :].float().cpu()
            base_ld = (logits_base[wrong_sign_tok] - logits_base[correct_sign_tok]).item()
            del out_base, logits_base

            entry = {
                "id":           qid,
                "domain":       domain,
                "written_sign": written_sign,
                "baseline_ld":  round(base_ld, 4),
            }
            for L in CAPTURE_LAYERS:
                if L in captured:
                    h_vec = captured[L]
                    entry[f"proj_d_L{L}"] = round(torch.dot(h_vec, d_hat).item(), 5)
                    entry[f"norm_h_L{L}"] = round(h_vec.norm().item(), 4)
                else:
                    entry[f"proj_d_L{L}"] = None
                    entry[f"norm_h_L{L}"] = None

            results.append(entry)
            del full_ids
            gc.collect(); torch.cuda.empty_cache()

    print(f"\n[expCapture] Total: {len(results)} entries")
    meta = {
        "n_entries": len(results),
        "domains": DOMAINS, "capture_layers": CAPTURE_LAYERS,
        "fields": "proj_d = h_L · d_hat (signed); norm_h = ||h_L||",
        "d_hat_recipe": "space-prefixed, norm-weighted",
        "purpose": "L78 error-modulated magnitude analysis (volume-knob claim)",
    }
    output = {"meta": meta, "data": results}
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=os.path.dirname(out_file),
                                     suffix=".tmp", delete=False) as f:
        json.dump(output, f, indent=2); tmp = f.name
    shutil.move(tmp, out_file)
    print(f"[expCapture] Saved → {out_file}")
    return results
