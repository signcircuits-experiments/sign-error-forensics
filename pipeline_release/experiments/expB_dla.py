"""
expB_dla.py
===========
Universal Direct Logit Attribution experiment.

RED-TEAM FIXES IMPLEMENTED:
  F1: contamination check (sign_char_offset > prefix_len)
  F3: wrong_sign_tok / correct_sign_tok naming
  F4: compute_eff_dir() in adapter (model-specific corrections)
  DLA-C: embedding contribution included (no longer omitted)
  Layer-level MLP + per-head attention DLA both stored
  Sanity check: |DLA| > 50 flagged as suspicious

Output JSON per question:
  {
    "id": str,
    "domain": "det" or "ibp",
    "wrong_sign_tok": int,
    "correct_sign_tok": int,
    "probe_tok": int,
    "embedding_dla": float,         # DLA-C fix: embedding now included
    "mlp_dla": {"0": float, ...},   # per-layer MLP DLA (all layers)
    "attn_dla": {"0": float, ...},  # per-layer attention DLA (all layers)
    "head_dla": {"L": [float*N_HEADS], ...},  # per-head DLA (selected layers)
    "top_mlp_layers": [[layer, dla], ...],    # top 5 by abs DLA
    "top_attn_heads": [[layer, head, dla], ...],  # top 10 by abs DLA
    "dla_sum": float,               # embedding + sum(mlp) + sum(attn)
    "suspicious": bool,             # True if any |DLA| > 50
  }
"""

import torch
import torch.nn.functional as F
import json
import os
import gc
import shutil
import tempfile
import pandas as pd
from adapters.base_adapter import BaseAdapter
from config import RESULTS_DIR

# Layers at which to store per-head DLA (expensive — selected layers only)
PER_HEAD_LAYERS = [60, 70, 75, 78, 79]   # adjust per model after expA results


def run_expB(adapter: BaseAdapter, df: pd.DataFrame,
             domain: str, out_file: str,
             per_head_layers: list = None):
    """
    Run DLA on all questions in df.
    per_head_layers: list of layer indices for per-head breakdown (None = PER_HEAD_LAYERS)
    """
    if per_head_layers is None:
        per_head_layers = PER_HEAD_LAYERS

    print(f"\n[expB] {adapter.model_name} | {domain} | {len(df)} questions")

    results = {}

    for _, row in df.iterrows():
        qid              = str(row["id"])
        wrong_sign       = str(row["wrong_sign"])
        correct_sign     = str(row["correct_sign"])
        sign_char_offset = int(row["sign_char_offset"])
        prefix_len       = int(row["prefix_len"])
        full_input       = str(row["full_input"])

        print(f"\n  [{qid}]")

        # ── F1: Contamination check ───────────────────────────────────────────
        if sign_char_offset <= prefix_len:
            print(f"  ⚠ CONTAMINATION — skipping")
            continue
        if full_input[sign_char_offset] != wrong_sign:
            print(f"  ⚠ CHAR MISMATCH — skipping")
            continue

        # ── Get token IDs and tokenize ────────────────────────────────────────
        try:
            wrong_sign_tok, correct_sign_tok = adapter.get_sign_token_ids(
                wrong_sign, sign_char_offset, full_input)
        except ValueError as e:
            print(f"  ⚠ {e} — skipping")
            continue

        full_ids = adapter.tokenize(full_input)
        tok_idx  = adapter.char_to_token_idx(sign_char_offset, full_input)
        probe_tok = tok_idx - 1

        if probe_tok < 1:
            print(f"  ⚠ probe_tok < 1 — skipping")
            continue

        # ── SINGLE forward pass: capture h_final + all MLP/Attn raw vectors ─────
        # Optimization: merge the former two passes (h_final capture + DLA hooks)
        # into one. Dot products are deferred until after the pass so eff_dir
        # (which depends on h_final) can be computed first.
        # This halves GPU time vs the two-pass approach on 70B+ models.
        #
        # Also fixes the double-norm bug: hook on last DECODER layer captures
        # the genuinely pre-final-RMSNorm residual stream state, not the
        # post-norm output from adapter.model.model() which caused 33+ logit unit
        # reconciliation errors.
        _raw_h_final = {}
        _raw_mlp     = {}   # {layer: h_vec}
        _raw_attn    = {}   # {layer: h_vec}
        _raw_head    = {}   # {layer: x_concat} for per-head layers

        def _cap_last(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if isinstance(h, torch.Tensor) and h.dim()==3 and h.shape[1]>probe_tok:
                _raw_h_final['h'] = h[0, probe_tok, :].detach().float().cpu()

        def _cap_mlp(L):
            def hook(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                if isinstance(h, torch.Tensor) and h.dim()==3 and h.shape[1]>probe_tok:
                    _raw_mlp[L] = h[0, probe_tok, :].detach().float().cpu()
            return hook

        def _cap_attn(L):
            def hook(module, inp, out):
                if isinstance(out, torch.Tensor) and out.dim()==3 and out.shape[1]>probe_tok:
                    _raw_attn[L] = out[0, probe_tok, :].detach().float().cpu()
                if L in per_head_layers and isinstance(inp, tuple) and inp[0] is not None:
                    _raw_head[L] = inp[0][0, probe_tok, :].detach().float().cpu()
            return hook

        # Register last-layer hook + all MLP + all attn hooks
        all_hooks = [adapter.get_layer_modules()[-1].register_forward_hook(_cap_last)]
        for L in range(adapter.N_LAYERS):
            all_hooks.append(adapter.get_mlp_module(L).register_forward_hook(_cap_mlp(L)))
            all_hooks.append(adapter.get_o_proj(L).register_forward_hook(_cap_attn(L)))

        with torch.no_grad():
            adapter.model.model(full_ids, use_cache=False)  # base model only — no lm_head

        for h in all_hooks:
            h.remove()

        if 'h' not in _raw_h_final:
            print(f"  ⚠ Failed to capture h_final — skipping")
            del full_ids; gc.collect(); torch.cuda.empty_cache()
            continue
        h_final = _raw_h_final['h']

        del full_ids
        gc.collect()
        torch.cuda.empty_cache()

        # ── Compute eff_dir from pre-norm h_final ─────────────────────────────
        eff_dir = adapter.compute_eff_dir(wrong_sign_tok, correct_sign_tok, h_final)

        # ── DLA-C: Embedding ──────────────────────────────────────────────────
        # Re-tokenize briefly just for embedding lookup (cheap — embed layer only)
        _full_ids_embed = adapter.tokenize(full_input)
        embed_h   = adapter.get_embedding_output(_full_ids_embed, probe_tok)
        embed_dla = float(torch.dot(embed_h, eff_dir))
        del _full_ids_embed

        # ── Deferred dot products: MLP and Attn DLA ───────────────────────────
        mlp_dlas  = {L: float(torch.dot(v, eff_dir)) for L, v in _raw_mlp.items()}
        attn_dlas = {L: float(torch.dot(v, eff_dir)) for L, v in _raw_attn.items()}

        # ── Deferred per-head DLA ─────────────────────────────────────────────
        head_dlas = {}
        for L, x_concat in _raw_head.items():
            W_o = adapter.get_o_proj(L).weight.detach().float().cpu()
            head_scores = []
            for h_idx in range(adapter.N_HEADS):
                s = h_idx * adapter.HEAD_DIM
                e = s + adapter.HEAD_DIM
                x_h      = x_concat[s:e]
                W_o_h    = W_o[:, s:e]
                head_out = W_o_h @ x_h
                head_scores.append(round(float(torch.dot(head_out, eff_dir)), 4))
            head_dlas[L] = head_scores
        gc.collect()
        torch.cuda.empty_cache()

        # ── Sanity check 1: magnitude outliers ───────────────────────────────
        all_vals = list(mlp_dlas.values()) + list(attn_dlas.values())
        suspicious = any(abs(v) > 50 for v in all_vals)
        if suspicious:
            print(f"  ⚠ Suspicious values (|DLA|>50) — possible architecture bug!")

        # ── Sanity check 2: DLA sum reconciliation ────────────────────────────
        # Compare dla_sum to the true logit_diff at h_final.
        # If eff_dir is correct, dla_sum ≈ true_logit_diff (within linearisation error).
        # Large discrepancy → wrong Jacobian or norm correction in compute_eff_dir.
        true_logit_diff = adapter.compute_logit_diff(
            h_final, wrong_sign_tok, correct_sign_tok)
        dla_sum_raw = embed_dla + sum(mlp_dlas.values()) + sum(attn_dlas.values())
        reconciliation_error = abs(dla_sum_raw - true_logit_diff)
        eff_dir_suspect = reconciliation_error > 0.5
        if eff_dir_suspect:
            print(f"  ⚠ eff_dir suspect: dla_sum={dla_sum_raw:.3f} but "
                  f"true_logit_diff={true_logit_diff:.3f} "
                  f"(error={reconciliation_error:.3f}) — check compute_eff_dir()")

        # ── Top contributors ──────────────────────────────────────────────────
        top_mlp = sorted(mlp_dlas.items(),
                         key=lambda x: abs(x[1]), reverse=True)[:5]
        top_mlp = [[int(L), round(v, 4)] for L, v in top_mlp]

        top_heads = []
        for L, scores in head_dlas.items():
            for h_idx, dla in enumerate(scores):
                top_heads.append([int(L), int(h_idx), dla])
        top_heads = sorted(top_heads, key=lambda x: abs(x[2]), reverse=True)[:10]

        dla_sum = embed_dla + sum(mlp_dlas.values()) + sum(attn_dlas.values())

        print(f"  embed_dla={embed_dla:.3f}  dla_sum={dla_sum_raw:.3f}  "
              f"true_logit_diff={true_logit_diff:.3f}  "
              f"recon_err={reconciliation_error:.3f}  suspicious={suspicious}")

        # ── Store ─────────────────────────────────────────────────────────────
        results[qid] = {
            "id"           : qid,
            "domain"       : domain,
            "sign_subtype" : str(row.get("sign_subtype", "")),
            "written_sign" : wrong_sign,
            "correct_sign" : correct_sign,
            "wrong_sign_tok"  : wrong_sign_tok,
            "correct_sign_tok" : correct_sign_tok,
            "probe_tok"    : probe_tok,
            "embedding_dla": round(embed_dla, 4),
            "mlp_dla"      : {str(L): round(v, 4) for L, v in mlp_dlas.items()},
            "attn_dla"     : {str(L): round(v, 4) for L, v in attn_dlas.items()},
            "head_dla"     : {str(L): scores for L, scores in head_dlas.items()},
            "top_mlp_layers" : top_mlp,
            "top_attn_heads" : top_heads,
            "true_logit_diff"       : round(true_logit_diff, 4),
            "dla_sum"               : round(dla_sum_raw, 4),
            "reconciliation_error"  : round(reconciliation_error, 4),
            "suspicious"            : suspicious,
            "eff_dir_suspect"       : eff_dir_suspect,
        }
        _safe_save(results, out_file)

    print(f"\n[expB] Done. {len(results)} questions saved.")
    return results


def _safe_save(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".tmp", delete=False) as f:
        json.dump(data, f, indent=2)
        tmp = f.name
    shutil.move(tmp, filepath)
