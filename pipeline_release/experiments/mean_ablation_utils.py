"""
mean_ablation_utils.py
======================
Shared utilities for expC', expD', expE' mean-ablation experiments.

DESIGN RATIONALE (pre-registered):
  Zero-ablation pushes activations out-of-distribution (OOD). Mean ablation
  replaces a component's output with its average over the 35 det error contexts,
  keeping downstream layers in-distribution. For per-position experiments we use
  the mean at the probe position only (one token before the sign), since that is
  the semantically consistent position across all questions.

MEAN COMPUTATION:
  For head mean (expC'): mean of head H's o_proj input slice
    [h*HEAD_DIM : (h+1)*HEAD_DIM] at probe_tok, averaged over the 35 det
    error contexts.
  For MLP mean (expD', expE'): mean of the MLP output vector at probe_tok,
    averaged over the 35 det error contexts.
  Both means are computed in a single pre-pass (no repeated forward passes).

SELF-REPAIR LOGGING (expC', expD'):
  After each ablation, we log the DLA change for 9 key components:
    MLPs at L66, L71, L72, L74, L75, L77, L78, L79
    Attn at L75
  DLA = dot(component_output, eff_dir). We capture before (baseline pass)
  and after (ablated pass) and store the difference as the self-repair vector.
"""

import torch
import json
import os
import gc
import math

# ── Key layers for self-repair logging ───────────────────────────────────────
SELF_REPAIR_MLP_LAYERS  = [66, 71, 72, 74, 75, 77, 78, 79]
SELF_REPAIR_ATTN_LAYERS = [75]


def compute_head_means(adapter, df, error_ids: set) -> dict:
    """
    Pre-pass: compute mean o_proj input slice for each head at L75 and L77,
    at the probe position, averaged over the error_ids questions.

    Returns: {layer_idx: tensor[N_HEADS, HEAD_DIM]} on CPU
    """
    target_layers = [75, 77]
    accum  = {L: None for L in target_layers}
    counts = {L: 0 for L in target_layers}

    raw_inputs = {L: [] for L in target_layers}

    for _, row in df.iterrows():
        qid = str(row['id'])
        if qid not in error_ids:
            continue

        wrong_sign       = str(row['wrong_sign'])
        sign_char_offset = int(row['sign_char_offset'])
        prefix_len       = int(row['prefix_len'])
        full_input       = str(row['full_input'])

        if sign_char_offset <= prefix_len: continue
        if full_input[sign_char_offset] != wrong_sign: continue

        full_ids  = adapter.tokenize(full_input)
        tok_idx   = adapter.char_to_token_idx(sign_char_offset, full_input)
        probe_tok = tok_idx - 1
        if probe_tok < 1:
            del full_ids; gc.collect(); continue

        captured = {}

        def make_hook(L):
            def hook(module, inp, out):
                if isinstance(inp, tuple) and inp[0] is not None:
                    x = inp[0]   # [batch, seq, n_heads*head_dim]
                    if x.shape[1] > probe_tok:
                        captured[L] = x[0, probe_tok, :].detach().float().cpu().clone()
            return hook

        handles = [adapter.get_o_proj(L).register_forward_hook(make_hook(L))
                   for L in target_layers]
        with torch.no_grad():
            adapter.model.model(full_ids, use_cache=False)
        for h in handles:
            h.remove()

        for L in target_layers:
            if L in captured:
                raw_inputs[L].append(captured[L])

        del full_ids; gc.collect(); torch.cuda.empty_cache()

    # Average
    means = {}
    for L in target_layers:
        if raw_inputs[L]:
            stacked = torch.stack(raw_inputs[L], dim=0)  # [N, n_heads*head_dim]
            means[L] = stacked.mean(dim=0)               # [n_heads*head_dim]
    return means


def compute_mlp_means(adapter, df, error_ids: set, target_layers: list) -> dict:
    """
    Pre-pass: compute mean MLP output at probe position for each target layer,
    averaged over error_ids questions.

    Returns: {layer_idx: tensor[hidden_dim]} on CPU
    """
    raw_outputs = {L: [] for L in target_layers}

    for _, row in df.iterrows():
        qid = str(row['id'])
        if qid not in error_ids:
            continue

        wrong_sign       = str(row['wrong_sign'])
        sign_char_offset = int(row['sign_char_offset'])
        prefix_len       = int(row['prefix_len'])
        full_input       = str(row['full_input'])

        if sign_char_offset <= prefix_len: continue
        if full_input[sign_char_offset] != wrong_sign: continue

        full_ids  = adapter.tokenize(full_input)
        tok_idx   = adapter.char_to_token_idx(sign_char_offset, full_input)
        probe_tok = tok_idx - 1
        if probe_tok < 1:
            del full_ids; gc.collect(); continue

        captured = {}

        def make_hook(L):
            def hook(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                if isinstance(h, torch.Tensor) and h.dim() == 3 and h.shape[1] > probe_tok:
                    captured[L] = h[0, probe_tok, :].detach().float().cpu().clone()
            return hook

        handles = [adapter.get_mlp_module(L).register_forward_hook(make_hook(L))
                   for L in target_layers]
        with torch.no_grad():
            adapter.model.model(full_ids, use_cache=False)
        for h in handles:
            h.remove()

        for L in target_layers:
            if L in captured:
                raw_outputs[L].append(captured[L])

        del full_ids; gc.collect(); torch.cuda.empty_cache()

    return {L: torch.stack(vs, dim=0).mean(dim=0) for L, vs in raw_outputs.items() if vs}


def run_baseline_pass(adapter, full_ids, probe_tok, wrong_sign_tok, correct_sign_tok,
                      capture_layers_mlp, capture_layers_attn):
    """
    Forward pass capturing: final logit_diff, MLP outputs, Attn outputs at
    the specified layers (for self-repair logging).
    Returns: (logit_diff, {L: mlp_out}, {L: attn_out})
    """
    captured_final = {}
    captured_mlp   = {}
    captured_attn  = {}

    def cap_last(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        if isinstance(h, torch.Tensor) and h.dim()==3 and h.shape[1]>probe_tok:
            captured_final['h'] = h[0, probe_tok, :].detach().float().cpu()

    def make_mlp_hook(L):
        def hook(m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            if isinstance(h, torch.Tensor) and h.dim()==3 and h.shape[1]>probe_tok:
                captured_mlp[L] = h[0, probe_tok, :].detach().float().cpu()
        return hook

    def make_attn_hook(L):
        def hook(m, i, o):
            if isinstance(o, torch.Tensor) and o.dim()==3 and o.shape[1]>probe_tok:
                captured_attn[L] = o[0, probe_tok, :].detach().float().cpu()
        return hook

    handles = [adapter.get_layer_modules()[-1].register_forward_hook(cap_last)]
    for L in capture_layers_mlp:
        handles.append(adapter.get_mlp_module(L).register_forward_hook(make_mlp_hook(L)))
    for L in capture_layers_attn:
        handles.append(adapter.get_o_proj(L).register_forward_hook(make_attn_hook(L)))

    with torch.no_grad():
        adapter.model.model(full_ids, use_cache=False)
    for h in handles:
        h.remove()

    logit_diff = 0.0
    if 'h' in captured_final:
        logit_diff = adapter.compute_logit_diff(
            captured_final['h'], wrong_sign_tok, correct_sign_tok)

    return logit_diff, captured_mlp, captured_attn


def compute_dla_from_captured(mlp_dict, attn_dict, eff_dir):
    """Compute DLA for each captured component."""
    result = {}
    for L, v in mlp_dict.items():
        result[f"mlp_L{L}"] = round(float(torch.dot(v, eff_dir)), 4)
    for L, v in attn_dict.items():
        result[f"attn_L{L}"] = round(float(torch.dot(v, eff_dir)), 4)
    return result


def safe_save(data, filepath):
    import shutil, tempfile
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=os.path.dirname(filepath),
                                     suffix=".tmp", delete=False) as f:
        json.dump(data, f, indent=2)
        tmp = f.name
    shutil.move(tmp, filepath)
