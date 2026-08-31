"""
expA_logit_lens.py
==================
Universal logit lens experiment.

RED-TEAM FIXES IMPLEMENTED:
  F1: sign_char_offset > prefix_len assertion (contamination filter)
  F2: ALL layers probed in ONE forward pass (hook on every layer)
  F3: signed argmax — peak_layer = argmax(logit_diff) [already signed correctly]
      wrong_sign_tok / correct_sign_tok naming (retired wrong_tok/correct_tok)
  F4: model-specific compute_logit_diff() in adapter (handles softcap, 1+w)
  F5: flip_layer with NOISE_FLOOR threshold (not first crossing at any prob)
      both peak_layer and flip_layer stored in JSON
  F7: version assertion in config.py (run before this script)

Output JSON per question:
  {
    "id": str,
    "domain": "det" or "ibp",
    "written_sign": "+" or "-",
    "correct_sign": "+" or "-",
    "wrong_sign_tok": int,
    "correct_sign_tok": int,
    "probe_tok": int,
    "n_tokens": int,
    "prefix_len": int,
    "contamination_ok": bool,       # sign_char_offset > prefix_len
    "peak_layer": int or null,      # argmax(logit_diff) above noise floor
    "peak_depth_pct": float,
    "peak_logit_diff": float,
    "flip_layer": int or null,      # first layer where logit_diff > 0, above noise floor
    "flip_depth_pct": float or null,
    "layer_diffs": {L: float, ...}, # logit_diff at every layer
  }
"""

import torch
import json
import os
import gc
import shutil
import tempfile
import pandas as pd
from adapters.base_adapter import BaseAdapter
from config import NOISE_FLOOR_LOGIT, RESULTS_DIR


def run_expA(adapter: BaseAdapter, df: pd.DataFrame,
             domain: str, out_file: str):
    """
    Run logit lens on all questions in df.

    adapter: loaded model adapter (model already on GPU)
    df: dataframe with columns: id, wrong_sign, correct_sign, sign_char_offset,
        prefix_len, full_input
    domain: "det" or "ibp"
    out_file: path to save results JSON
    """
    print(f"\n[expA] {adapter.model_name} | {domain} | {len(df)} questions")
    print(f"[expA] Probing ALL {adapter.N_LAYERS} layers per question")

    results = {}
    n_contaminated = 0

    for _, row in df.iterrows():
        qid             = str(row["id"])
        wrong_sign      = str(row["wrong_sign"])
        correct_sign    = str(row["correct_sign"])
        sign_char_offset = int(row["sign_char_offset"])
        prefix_len      = int(row["prefix_len"])
        full_input      = str(row["full_input"])

        print(f"\n  [{qid}] written='{wrong_sign}' correct='{correct_sign}' "
              f"offset={sign_char_offset} prefix={prefix_len}")

        # ── F1: Contamination check ───────────────────────────────────────────
        contamination_ok = sign_char_offset > prefix_len
        if not contamination_ok:
            n_contaminated += 1
            print(f"  ⚠ CONTAMINATION: offset {sign_char_offset} <= prefix {prefix_len} — skipping")
            continue

        # Character at offset must be wrong_sign
        if full_input[sign_char_offset] != wrong_sign:
            print(f"  ⚠ CHAR MISMATCH: full_input[{sign_char_offset}]="
                  f"'{full_input[sign_char_offset]}' != '{wrong_sign}' — skipping")
            continue

        # ── Get token IDs ─────────────────────────────────────────────────────
        try:
            wrong_sign_tok, correct_sign_tok = adapter.get_sign_token_ids(
                wrong_sign, sign_char_offset, full_input
            )
        except ValueError as e:
            print(f"  ⚠ Token ID error: {e} — skipping")
            continue

        # ── Tokenize and find probe position ─────────────────────────────────
        full_ids = adapter.tokenize(full_input)
        n_tokens = full_ids.shape[1]

        tok_idx = adapter.char_to_token_idx(sign_char_offset, full_input)
        probe_tok = tok_idx - 1   # hidden state before the sign token

        if probe_tok < 1:
            print(f"  ⚠ probe_tok={probe_tok} < 1 — skipping "
                  f"(sign is at token index {tok_idx}, need at least tok_idx=1)")
            continue

        print(f"  tokens={n_tokens}  probe_tok={probe_tok}  "
              f"wrong_sign_tok={wrong_sign_tok}  correct_sign_tok={correct_sign_tok}")

        # ── F2: Hook ALL layers, ONE forward pass ─────────────────────────────
        layer_diffs = {}   # {layer_int: logit_diff_float}
        hooks = []

        def make_hook(layer_idx):
            def hook(module, inp, out):
                # out is tuple (hidden_states, ...) or just tensor
                hidden = out[0] if isinstance(out, tuple) else out
                if not isinstance(hidden, torch.Tensor) or hidden.dim() != 3:
                    return
                if hidden.shape[1] <= probe_tok:
                    return
                h = hidden[0, probe_tok, :].detach().float().cpu().clone()
                diff = adapter.compute_logit_diff(h, wrong_sign_tok, correct_sign_tok)
                layer_diffs[layer_idx] = diff
            return hook

        for L in range(adapter.N_LAYERS):
            h = adapter.get_layer_modules()[L].register_forward_hook(make_hook(L))
            hooks.append(h)

        with torch.no_grad():
            adapter.model(full_ids, use_cache=False)

        for h in hooks:
            h.remove()
        del full_ids
        gc.collect()
        torch.cuda.empty_cache()

        if not layer_diffs:
            print(f"  ⚠ No layers captured — skipping")
            continue

        # ── F3 + F5: Signed peak_layer and flip_layer with noise floor ────────
        # Noise floor on raw logit_diff magnitude:
        #   |logit_diff| < NOISE_FLOOR_LOGIT → model has no meaningful preference
        #   between the two sign tokens at this layer → skip for flip_layer.
        # NOTE: sigmoid(logit_diff) ≥ 0.5 always, so a sigmoid-based gate
        #   can never filter anything — use raw magnitude instead.
        # NOISE_FLOOR_LOGIT imported from config — single source of truth

        peak_layer    = None
        peak_diff     = None
        flip_layer    = None

        for L in sorted(layer_diffs.keys()):
            diff = layer_diffs[L]
            # Peak: highest logit_diff across all layers (no noise gate — we want
            #   the true maximum even if it's small; gating would bias toward late layers)
            if peak_diff is None or diff > peak_diff:
                peak_diff  = diff
                peak_layer = L
            # Flip: first layer above noise floor where model prefers wrong sign
            if flip_layer is None and diff > NOISE_FLOOR_LOGIT:  # from config
                flip_layer = L

        peak_depth_pct = (peak_layer / adapter.N_LAYERS * 100) if peak_layer is not None else None
        flip_depth_pct = (flip_layer / adapter.N_LAYERS * 100) if flip_layer is not None else None

        pct_str = f"{peak_depth_pct:.1f}%" if peak_depth_pct is not None else "None"
        print(f"  peak_layer={peak_layer} ({pct_str})  "
              f"peak_diff={peak_diff:.3f}  flip_layer={flip_layer}")

        # ── Store result ──────────────────────────────────────────────────────
        results[qid] = {
            "id"              : qid,
            "domain"          : domain,
            "sign_subtype"    : str(row.get("sign_subtype", "")),
            "written_sign"    : wrong_sign,
            "correct_sign"    : correct_sign,
            "wrong_sign_tok"  : wrong_sign_tok,
            "correct_sign_tok": correct_sign_tok,
            "probe_tok"       : probe_tok,
            "n_tokens"        : n_tokens,
            "prefix_len"      : prefix_len,
            "contamination_ok": contamination_ok,
            "peak_layer"      : peak_layer,
            # BUG FIX: use 'is not None' not truthiness — peak_depth_pct=0.0 is valid
            "peak_depth_pct"  : round(peak_depth_pct, 2) if peak_depth_pct is not None else None,
            "peak_logit_diff" : round(peak_diff, 4) if peak_diff is not None else None,
            "flip_layer"      : flip_layer,
            "flip_depth_pct"  : round(flip_depth_pct, 2) if flip_depth_pct is not None else None,
            "layer_diffs"     : {str(L): round(v, 4) for L, v in layer_diffs.items()},
        }

        # Checkpoint after each question
        _safe_save(results, out_file)

    print(f"\n[expA] Done. {len(results)} questions saved. "
          f"{n_contaminated} contaminated (skipped).")
    if n_contaminated > 0:
        print(f"[expA] ⚠ {n_contaminated} questions had sign in prompt — check data!")

    return results


def _safe_save(data: dict, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".tmp", delete=False) as f:
        json.dump(data, f, indent=2)
        tmp = f.name
    shutil.move(tmp, filepath)
