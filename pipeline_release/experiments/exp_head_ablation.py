"""
exp_head_ablation.py — Generalized per-head mean-ablation (EXPLORATORY)
========================================================================

*** EXPLORATORY — NOT part of the pre-registered experiment set for the 3
new small models (gemma-3-27b-it, phi-4, mistral-small-3.2-24b-instruct).
Head ablation for these models is a follow-up/exploratory analysis; unlike
expC_prime.py's Llama-specific pre-registered L75/L77 targets (chosen from
prior expB DLA results on Llama), there is no pre-registered target list
for the new models — targets must be supplied explicitly via --sites. ***

WHY THIS FILE EXISTS (vs. reusing expC_prime.py):
  expC_prime.py hardcodes:
    - L77_TARGETS / L75 target-selection logic specific to Llama's det
      domain circuit (heads discovered on Llama-3.3-70B).
    - SELF_REPAIR_MLP_LAYERS / SELF_REPAIR_ATTN_LAYERS from
      mean_ablation_utils.py, which are also Llama-specific hardcoded layer
      lists (checked: [66, 71, 72, 74, 75, 77, 78, 79] / [75] — these are
      the layers where Llama's det-domain circuit was found in prior
      analysis; they have NO meaning for Gemma/Phi/Mistral's (as-yet
      unknown) circuits).
    - `adapter.model.model(...)` and `adapter.get_o_proj(L)` are already
      adapter-routed in expC_prime.py, which is good — but the TARGET
      layers/heads and self-repair layers are not, and would need a full
      rewrite per model. Rather than hardcode a second model-specific
      target list, this module takes targets from a JSON file supplied at
      run time (--sites), keeping the code model- and target-agnostic.

WHAT'S SIMPLIFIED vs. expC_prime.py (documented per Part 3 instructions):
  - Self-repair matrix logging is SIMPLIFIED: instead of a fixed
    Llama-specific 9-component self-repair set (8 MLP layers + 1 attn
    layer, hardcoded), this module optionally logs self-repair DLA at a
    small set of `self_repair_layers` passed in by the caller (default:
    empty list, i.e. self-repair logging is OFF). Rationale: for 3 new
    models with completely unknown circuits, there is no principled way to
    pick "the 9 key layers" without an expB'-equivalent pass first; keeping
    this optional avoids inventing a fake pre-registered list.
  - Only ONE ablation method is implemented per run: MEAN ablation (matching
    expC_prime.py's default), with a `--zero-ablation-fallback` flag that
    switches to zero-ablation instead (useful if the mean-computation pass
    itself is what's under test, or if there aren't enough contexts to
    compute a stable mean — e.g. very small IBP sets like n=12/15).
  - No "group" (embedding_bias vs late_circuit) classification — that
    required expA_logit_lens.json's peak_layer field and Llama-specific
    layer-depth intuition; the raw baseline_ld / edit_ld / delta_ld numbers
    are logged per site per case and grouping is left to downstream
    analysis, which works off the output JSON.

CRITICAL ADAPTER-ONLY RULE:
  ALL module access below goes through the adapter interface
  (adapter.get_o_proj(L), adapter.get_layer_modules(), adapter.tokenize(),
  adapter.char_to_token_idx(), adapter.compute_logit_diff(),
  adapter.get_sign_token_ids()) — there is NO hardcoded module path
  (e.g. no `model.model.layers[L].self_attn.o_proj` literal) anywhere in
  this file. This directly addresses the "TP-sharding hook failure on
  Mistral-large" class of bug cited in the project notes, which was caused
  by exactly this kind of hardcoding.

SITES FILE SCHEMA (--sites path/to/sites.json):
  [
    {"layer": 40, "head": 12, "role": "pusher",    "name": "L40_H12_push"},
    {"layer": 40, "head": 3,  "role": "corrector", "name": "L40_H3_corr"},
    {"layer": 40, "head": 7,  "role": "control",   "name": "L40_H7_ctrl"},
    ...
  ]
  "role" in {"pusher", "corrector", "control", "exploratory"}.

OUTPUT SCHEMA (matches expD'-style, minimum required fields):
  {qid}: {
    "id": ..., "domain": ..., "baseline_ld": float,
    "sites": {
      site_name: {
        "layer": int, "head": int, "role": str,
        "baseline_ld": float, "edit_ld": float, "delta_ld": float,
        "self_repair": {component_key: delta_dla} or {}   # optional, see above
      }, ...
    }
  }

DRY-RUN MODE:
  run_dry_run(adapter, sites) resolves adapter.get_o_proj(L) for every site,
  registers a no-op forward hook, runs NOTHING (no forward pass), then
  immediately removes the hook and reports what it resolved: module class
  name, in_features/out_features (if a Linear), and whether head*head_dim
  fits inside the module's input dimension. This is meant to catch a wrong
  module path (e.g. Phi's assumed-unfused o_proj, see phi_adapter.py
  docstring) BEFORE any GPU time is spent — directly addresses the
  "960/960 no-capture" failure class mentioned in the project notes (a
  hook silently registered on the wrong/nonexistent tensor shape captures
  nothing, and the bug is invisible until someone notices 0 useful rows in
  the output JSON).
"""

import gc
import json
import os

import pandas as pd
import torch

from adapters.base_adapter import BaseAdapter
from experiments.mean_ablation_utils import compute_dla_from_captured, safe_save


# ── Dry-run (no forward pass, no GPU work beyond hook (de)registration) ──────

def run_dry_run(adapter: "BaseAdapter", sites: list) -> dict:
    """
    Resolve every site's o_proj module via the adapter, register+immediately
    remove a no-op hook, and report what was resolved. Does NOT run any
    forward pass. Returns a report dict; also prints a human-readable
    summary. Call this before any real ablation run on a new model.
    """
    report = {"n_sites": len(sites), "resolved": [], "failed": []}

    for site in sites:
        layer = site["layer"]
        head = site["head"]
        name = site.get("name", f"L{layer}_H{head}")
        entry = {"name": name, "layer": layer, "head": head}
        try:
            module = adapter.get_o_proj(layer)
            entry["module_class"] = type(module).__name__

            in_features = getattr(module, "in_features", None)
            out_features = getattr(module, "out_features", None)
            entry["in_features"] = in_features
            entry["out_features"] = out_features

            head_dim = adapter.HEAD_DIM
            n_heads = adapter.N_HEADS
            expected_in = n_heads * head_dim
            entry["expected_in_features"] = expected_in
            entry["in_features_match"] = (in_features == expected_in) if in_features else None

            s = head * head_dim
            e = s + head_dim
            entry["head_slice"] = [s, e]
            entry["head_slice_in_bounds"] = (in_features is not None and e <= in_features)

            # Register + immediately remove a no-op hook — verifies the
            # module object supports register_forward_hook (i.e. it is a
            # real nn.Module attached to the live model, not e.g. None or a
            # stray attribute), without running any data through the model.
            captured = {}

            def _noop_hook(m, inp, out, _captured=captured):
                _captured["called"] = True
                return out

            handle = module.register_forward_hook(_noop_hook)
            handle.remove()
            entry["hook_registerable"] = True

            entry["status"] = "OK" if entry["in_features_match"] and entry["head_slice_in_bounds"] else "SUSPECT"
            report["resolved"].append(entry)
        except Exception as e:
            entry["status"] = "FAILED"
            entry["error"] = str(e)
            report["failed"].append(entry)

    print(f"\n[exp_head_ablation dry-run] {adapter.model_name}: "
          f"{len(report['resolved'])} resolved, {len(report['failed'])} failed "
          f"out of {report['n_sites']} sites")
    for entry in report["resolved"]:
        flag = "" if entry["status"] == "OK" else "  <-- SUSPECT, verify manually"
        print(f"  [{entry['status']:7s}] {entry['name']:16s} "
              f"L{entry['layer']:>3} H{entry['head']:>3}  "
              f"module={entry['module_class']}  "
              f"in_features={entry['in_features']} (expected {entry['expected_in_features']})"
              f"{flag}")
    for entry in report["failed"]:
        print(f"  [FAILED ] {entry['name']:16s} L{entry['layer']:>3} H{entry['head']:>3}  "
              f"error={entry['error']}")

    return report


# ── Mean-activation pre-pass (per layer, reused across all heads in that layer) ──

def _compute_layer_means(adapter, df: pd.DataFrame, layers: list) -> dict:
    """
    Pre-pass: mean o_proj INPUT (pre-o_proj, i.e. concatenated per-head
    values) at the probe position, averaged over all rows in df, for each
    layer in `layers`. Returns {layer: tensor[n_heads*head_dim]} (CPU).

    Adapter-only access: uses adapter.get_o_proj(L), adapter.tokenize(),
    adapter.char_to_token_idx() exclusively — no hardcoded module paths.
    """
    accum = {L: [] for L in layers}

    for _, row in df.iterrows():
        wrong_sign = str(row["wrong_sign"])
        sign_char_offset = int(row["sign_char_offset"])
        prefix_len = int(row["prefix_len"])
        full_input = str(row["full_input"])

        if sign_char_offset <= prefix_len:
            continue
        if full_input[sign_char_offset] != wrong_sign:
            continue

        full_ids = adapter.tokenize(full_input)
        tok_idx = adapter.char_to_token_idx(sign_char_offset, full_input)
        probe_tok = tok_idx - 1
        if probe_tok < 1:
            del full_ids
            gc.collect()
            continue

        captured = {}

        def make_hook(L):
            def hook(module, inp, out):
                if isinstance(inp, tuple) and inp[0] is not None:
                    x = inp[0]
                    if x.shape[1] > probe_tok:
                        captured[L] = x[0, probe_tok, :].detach().float().cpu().clone()
            return hook

        handles = [adapter.get_o_proj(L).register_forward_hook(make_hook(L)) for L in layers]
        with torch.no_grad():
            adapter.model.model(full_ids, use_cache=False)
        for h in handles:
            h.remove()

        for L in layers:
            if L in captured:
                accum[L].append(captured[L])

        del full_ids
        gc.collect()
        torch.cuda.empty_cache()

    means = {}
    for L in layers:
        if accum[L]:
            means[L] = torch.stack(accum[L], dim=0).mean(dim=0)
    return means


# ── Main ablation run ────────────────────────────────────────────────────────

def run_head_ablation(adapter: "BaseAdapter", df: pd.DataFrame, domain: str,
                       out_file: str, sites: list,
                       zero_ablation_fallback: bool = False,
                       self_repair_layers: list = None):
    """
    Per-head mean-ablation (or zero-ablation if zero_ablation_fallback=True)
    at the o_proj input, probe position only. All module access via adapter.

    sites: list of {"layer": int, "head": int, "role": str, "name": str}
           (schema in module docstring).
    self_repair_layers: optional list of MLP layer indices to log Δ-DLA for
        after each ablation (simplified vs. expC_prime.py's fixed 9-component
        matrix — see module docstring). Empty/None disables self-repair
        logging entirely.
    """
    self_repair_layers = self_repair_layers or []

    layers_needed = sorted({s["layer"] for s in sites})
    print(f"\n[exp_head_ablation] {adapter.model_name} | {domain} | {len(df)} questions | "
          f"{len(sites)} sites across layers {layers_needed} | "
          f"method={'zero' if zero_ablation_fallback else 'mean'}-ablation | "
          f"self_repair_layers={self_repair_layers}")

    layer_means = {}
    if not zero_ablation_fallback:
        print(f"[exp_head_ablation] Computing per-layer mean o_proj inputs over "
              f"{len(df)} contexts...")
        layer_means = _compute_layer_means(adapter, df, layers_needed)
        missing = [L for L in layers_needed if L not in layer_means]
        if missing:
            print(f"[exp_head_ablation] WARNING: no mean available for layers "
                  f"{missing} (0 valid contexts captured) — sites at these "
                  f"layers will be skipped. Consider --zero-ablation-fallback.")

    results = {}

    for _, row in df.iterrows():
        qid = str(row["id"])
        wrong_sign = str(row["wrong_sign"])
        correct_sign = str(row["correct_sign"])
        sign_char_offset = int(row["sign_char_offset"])
        prefix_len = int(row["prefix_len"])
        full_input = str(row["full_input"])

        if sign_char_offset <= prefix_len:
            print(f"  [{qid}] CONTAMINATION — skipping")
            continue
        if full_input[sign_char_offset] != wrong_sign:
            print(f"  [{qid}] CHAR MISMATCH — skipping")
            continue

        try:
            wrong_sign_tok, correct_sign_tok = adapter.get_sign_token_ids(
                wrong_sign, sign_char_offset, full_input)
        except ValueError as e:
            print(f"  [{qid}] {e} — skipping")
            continue

        full_ids = adapter.tokenize(full_input)
        tok_idx = adapter.char_to_token_idx(sign_char_offset, full_input)
        probe_tok = tok_idx - 1
        if probe_tok < 1:
            print(f"  [{qid}] probe_tok < 1 — skipping")
            del full_ids
            continue

        # ── Baseline pass ──────────────────────────────────────────────────
        h_final_c = {}
        base_mlp_c = {}

        def cap_last_base(m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            if isinstance(h, torch.Tensor) and h.dim() == 3 and h.shape[1] > probe_tok:
                h_final_c["h"] = h[0, probe_tok, :].detach().float().cpu()

        def make_mlp_base(LL):
            def hook(m, i, o):
                h = o[0] if isinstance(o, tuple) else o
                if isinstance(h, torch.Tensor) and h.dim() == 3 and h.shape[1] > probe_tok:
                    base_mlp_c[LL] = h[0, probe_tok, :].detach().float().cpu()
            return hook

        base_handles = [adapter.get_layer_modules()[-1].register_forward_hook(cap_last_base)]
        for LL in self_repair_layers:
            base_handles.append(adapter.get_mlp_module(LL).register_forward_hook(make_mlp_base(LL)))

        with torch.no_grad():
            adapter.model.model(full_ids, use_cache=False)
        for hh in base_handles:
            hh.remove()

        if "h" not in h_final_c:
            print(f"  [{qid}] h_final capture failed — skipping")
            del full_ids
            gc.collect()
            continue

        eff_dir = adapter.compute_eff_dir(wrong_sign_tok, correct_sign_tok, h_final_c["h"])
        baseline_ld = adapter.compute_logit_diff(h_final_c["h"], wrong_sign_tok, correct_sign_tok)
        baseline_dla = compute_dla_from_captured(base_mlp_c, {}, eff_dir) if self_repair_layers else {}

        site_results = {}

        for site in sites:
            L, H = site["layer"], site["head"]
            name = site.get("name", f"L{L}_H{H}")

            if not zero_ablation_fallback and L not in layer_means:
                continue

            head_dim = adapter.HEAD_DIM
            s = H * head_dim
            e = s + head_dim
            if zero_ablation_fallback:
                repl = torch.zeros(head_dim)
            else:
                repl = layer_means[L][s:e]

            edit_c = {}
            edit_mlp_c = {}

            def abl_hook(module, inp, out, _s=s, _e=e, _repl=repl):
                if isinstance(inp, tuple) and inp[0] is not None:
                    x = inp[0].clone()
                    if x.shape[1] > probe_tok:
                        x[0, probe_tok, _s:_e] = _repl.to(x.device, x.dtype)
                    result = torch.nn.functional.linear(
                        x, module.weight,
                        module.bias if module.bias is not None else None)
                    return result

            def cap_last_edit(m, i, o):
                h = o[0] if isinstance(o, tuple) else o
                if isinstance(h, torch.Tensor) and h.dim() == 3 and h.shape[1] > probe_tok:
                    edit_c["final"] = h[0, probe_tok, :].detach().float().cpu()

            def make_mlp_edit(LL):
                def hook(m, i, o):
                    h = o[0] if isinstance(o, tuple) else o
                    if isinstance(h, torch.Tensor) and h.dim() == 3 and h.shape[1] > probe_tok:
                        edit_mlp_c[LL] = h[0, probe_tok, :].detach().float().cpu()
                return hook

            handles = [
                adapter.get_o_proj(L).register_forward_hook(abl_hook),
                adapter.get_layer_modules()[-1].register_forward_hook(cap_last_edit),
            ]
            for LL in self_repair_layers:
                handles.append(adapter.get_mlp_module(LL).register_forward_hook(make_mlp_edit(LL)))

            with torch.no_grad():
                adapter.model.model(full_ids, use_cache=False)
            for hh in handles:
                hh.remove()

            edit_ld = 0.0
            if "final" in edit_c:
                edit_ld = adapter.compute_logit_diff(edit_c["final"], wrong_sign_tok, correct_sign_tok)
            delta_ld = edit_ld - baseline_ld

            self_repair = {}
            if self_repair_layers:
                edit_dla = compute_dla_from_captured(edit_mlp_c, {}, eff_dir)
                self_repair = {k: round(edit_dla.get(k, 0) - baseline_dla.get(k, 0), 4)
                                for k in set(list(edit_dla) + list(baseline_dla))}

            print(f"    [{qid}] {name} (L{L}_H{H}, {site.get('role', '?')}): "
                  f"delta_ld={delta_ld:+.4f}")

            site_results[name] = {
                "layer": L, "head": H, "role": site.get("role", "exploratory"),
                "baseline_ld": round(baseline_ld, 4),
                "edit_ld": round(edit_ld, 4),
                "delta_ld": round(delta_ld, 4),
                "self_repair": self_repair,
            }

        results[qid] = {
            "id": qid, "domain": domain,
            "baseline_ld": round(baseline_ld, 4),
            "is_error_case": baseline_ld > 0,
            "sites": site_results,
        }
        safe_save(results, out_file)

        del full_ids
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n[exp_head_ablation] Done. {len(results)} questions saved to {out_file}")
    return results
