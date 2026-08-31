"""
expE_prime.py — Joint MLP mean-ablation vs random-set null
==========================================================

PRE-REGISTERED DESIGN:

  Target set (from expA peak-layer distribution, pre-registered):
    [67, 71, 72, 74, 77]  — the 5 dominant late-circuit MLP layers

  Optional variant: add L75 attention ablation jointly (one extra run).

  Null distribution:
    40 random 5-layer MLP sets drawn from L50–L79, excluding the target set.
    Seeds: 0–39 (fixed, pre-registered).
    Provides an empirical null for "what does removing any 5 late layers do?"

  Criterion (REPLACES the failed pre-registered threshold from expE):
    Target-set Δlogit_diff more negative than the MINIMUM of all 40 null means
    (NULL_PERCENTILE = 0 = index 0 of sorted nulls).
    p = 1/(40+1) ≈ 0.024 one-sided (target beats all 40 random sets).
    Same criterion applied to ibp-10 and embedding-bias-12 separately:
      If ibp or emb are ALSO below 5th percentile → effect is generic late damage.
      If only det-late-circuit is below 5th percentile → circuit-specific.

  Method: MEAN ablation (same mean vectors as expD').
    Joint ablation: all target layers patched simultaneously in one forward pass.
    Random sets: same mean-ablation applied to each random set.

Output: {RESULTS_DIR}/{model}/{domain}/expE_prime_joint_mean_ablation.json
  Contains: target results per question + null distribution summary.
"""

import torch
import json
import os
import gc
import random
import statistics
import pandas as pd
from adapters.base_adapter import BaseAdapter
from config import RESULTS_DIR
from experiments.mean_ablation_utils import compute_mlp_means, safe_save

# ── PRE-REGISTERED CONSTANTS ──────────────────────────────────────────────────
TARGET_SET    = [67, 71, 72, 74, 77]
NULL_SEEDS    = list(range(40))           # seeds 0–39 (40 nulls → p<0.05 = beat all 40)
CANDIDATE_POOL = list(range(50, 80))      # L50–L79
NULL_SET_SIZE  = 5

# Pre-registered criterion
# With 40 nulls: beat all 40 = rank 0 = p = 1/41 ≈ 0.024 (one-sided)
# sorted(null_deltas)[0] = minimum of 40 nulls; target must be below that.
NULL_PERCENTILE = 0   # index 0 = must beat ALL 40 nulls (p ≈ 0.024)


def _draw_null_sets(candidate_pool, target_set, n_sets, seeds):
    """Draw n_sets random 5-layer sets from candidate_pool, excluding target_set."""
    excluded = set(target_set)
    pool = [L for L in candidate_pool if L not in excluded]
    sets = []
    for seed in seeds[:n_sets]:
        rng = random.Random(seed)
        s = sorted(rng.sample(pool, NULL_SET_SIZE))
        sets.append(s)
    return sets


def _ablate_joint(adapter, full_ids, probe_tok, wrong_sign_tok, correct_sign_tok,
                  layers_to_ablate, mlp_means):
    """Joint mean-ablation of multiple MLP layers, measure final logit_diff."""
    captured = {}

    def make_abl_hook(L, mean_v):
        def hook(m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            if isinstance(h, torch.Tensor) and h.dim()==3 and h.shape[1]>probe_tok:
                patched = h.clone()
                patched[0, probe_tok, :] = mean_v.to(patched.device)
                return (patched,) if isinstance(o, tuple) else patched
        return hook

    def cap_last(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        if isinstance(h, torch.Tensor) and h.dim()==3 and h.shape[1]>probe_tok:
            captured['h'] = h[0, probe_tok, :].detach().float().cpu()

    handles = [adapter.get_layer_modules()[-1].register_forward_hook(cap_last)]
    for L in layers_to_ablate:
        if L in mlp_means:
            handles.append(adapter.get_mlp_module(L).register_forward_hook(
                make_abl_hook(L, mlp_means[L])))

    with torch.no_grad():
        adapter.model.model(full_ids, use_cache=False)
    for h in handles: h.remove()

    if 'h' not in captured:
        return 0.0
    return adapter.compute_logit_diff(captured['h'], wrong_sign_tok, correct_sign_tok)


def run_expE_prime(adapter: BaseAdapter, df: pd.DataFrame,
                   domain: str, out_file: str,
                   expA_data: dict):
    """Run joint mean-ablation + random-set null (expE')."""

    error_ids = {qid for qid, q in expA_data.items() if q.get("peak_layer", 0) > 5}
    null_sets  = _draw_null_sets(CANDIDATE_POOL, TARGET_SET, len(NULL_SEEDS), NULL_SEEDS)
    all_layers_needed = list(set(TARGET_SET + [L for s in null_sets for L in s]))

    print(f"\n[expE'] {adapter.model_name} | {domain} | {len(df)} questions")
    print(f"[expE'] Target: {TARGET_SET}")
    print(f"[expE'] {len(null_sets)} random null sets (seeds 0–{len(null_sets)-1})")
    print(f"[expE'] Computing MLP means...")

    mlp_means = compute_mlp_means(adapter, df, error_ids, all_layers_needed)
    print(f"[expE'] Means ready for {len(mlp_means)} layers")

    results   = {}
    per_q_target = []    # for criterion computation
    per_q_nulls  = {i: [] for i in range(len(null_sets))}

    for _, row in df.iterrows():
        qid              = str(row["id"])
        wrong_sign       = str(row["wrong_sign"])
        correct_sign     = str(row["correct_sign"])
        sign_char_offset = int(row["sign_char_offset"])
        prefix_len       = int(row["prefix_len"])
        full_input       = str(row["full_input"])

        print(f"\n  [{qid}]")

        if sign_char_offset <= prefix_len:
            print(f"  CONTAMINATION — skipping"); continue
        if full_input[sign_char_offset] != wrong_sign:
            print(f"  CHAR MISMATCH — skipping"); continue

        try:
            wrong_sign_tok, correct_sign_tok = adapter.get_sign_token_ids(
                wrong_sign, sign_char_offset, full_input)
        except ValueError as e:
            print(f"  ⚠ {e} — skipping"); continue

        full_ids  = adapter.tokenize(full_input)
        tok_idx   = adapter.char_to_token_idx(sign_char_offset, full_input)
        probe_tok = tok_idx - 1
        if probe_tok < 1:
            print(f"  ⚠ probe_tok < 1"); del full_ids; continue

        # Baseline
        baseline_c = {}
        def cap_base(m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            if isinstance(h, torch.Tensor) and h.dim()==3 and h.shape[1]>probe_tok:
                baseline_c['h'] = h[0, probe_tok, :].detach().float().cpu()
        hh = adapter.get_layer_modules()[-1].register_forward_hook(cap_base)
        with torch.no_grad(): adapter.model.model(full_ids, use_cache=False)
        hh.remove()
        if 'h' not in baseline_c:
            del full_ids; continue
        baseline_ld = adapter.compute_logit_diff(
            baseline_c['h'], wrong_sign_tok, correct_sign_tok)

        # Target ablation
        target_ld = _ablate_joint(adapter, full_ids, probe_tok,
                                   wrong_sign_tok, correct_sign_tok,
                                   TARGET_SET, mlp_means)
        target_delta = target_ld - baseline_ld
        print(f"  baseline={baseline_ld:+.4f}  target_delta={target_delta:+.4f}", end="  ")

        # Null ablations
        null_deltas = []
        for i, null_set in enumerate(null_sets):
            null_ld = _ablate_joint(adapter, full_ids, probe_tok,
                                     wrong_sign_tok, correct_sign_tok,
                                     null_set, mlp_means)
            null_deltas.append(round(null_ld - baseline_ld, 4))

        null_p5 = sorted(null_deltas)[NULL_PERCENTILE]  # min of nulls (index 0)
        beats_null = target_delta < null_p5
        print(f"null_p5={null_p5:+.4f}  beats={'✓' if beats_null else '✗'}")

        expA_q = expA_data.get(qid, {})
        peak_layer = expA_q.get("peak_layer")
        group = ("embedding_bias" if peak_layer is not None and peak_layer <= 5
                 else "late_circuit" if peak_layer else "unknown")

        results[qid] = {
            "id":           qid,
            "domain":       domain,
            "group":        group,
            "peak_layer":   peak_layer,
            "baseline_ld":  round(baseline_ld, 4),
            "is_error_case": baseline_ld > 0,
            "target_delta": round(target_delta, 4),
            "null_deltas":  null_deltas,
            "null_p5":      round(null_p5, 4),
            "beats_null_p5": beats_null,
        }

        per_q_target.append(target_delta)
        for i, nd in enumerate(null_deltas):
            per_q_nulls[i].append(nd)

        safe_save(results, out_file)
        del full_ids; gc.collect(); torch.cuda.empty_cache()

    # ── Summary + criterion check ─────────────────────────────────────────────
    if per_q_target:
        mean_target = sum(per_q_target) / len(per_q_target)
        null_means  = [sum(vs)/len(vs) for vs in per_q_nulls.values() if vs]
        null_mean_p5 = sorted(null_means)[NULL_PERCENTILE] if null_means else None  # index 0 = min of 40
        pct_beats = sum(1 for q in results.values() if q.get("beats_null_p5")) / len(results)

        print(f"\n[expE'] SUMMARY")
        print(f"  Mean target delta:      {mean_target:+.4f}")
        print(f"  Null mean 5th pct:      {null_mean_p5:+.4f}")
        print(f"  Target beats null p5:   {mean_target < null_mean_p5 if null_mean_p5 else '?'}")
        print(f"  % questions beats null: {pct_beats:.1%}")

        # Write summary into output
        # Per-group summaries (pre-registered criterion applies per group)
        def group_summary(group_name):
            qs = [q for q in results.values()
                  if isinstance(q, dict) and q.get("group") == group_name]
            if not qs: return None
            tds = [q["target_delta"] for q in qs]
            nd_by_set = {}
            for q in qs:
                for i, nd in enumerate(q["null_deltas"]):
                    nd_by_set.setdefault(i, []).append(nd)
            null_grp_means = [sum(vs)/len(vs) for vs in nd_by_set.values() if vs]
            null_grp_p = sorted(null_grp_means)[NULL_PERCENTILE] if null_grp_means else None
            m = sum(tds)/len(tds)
            return {
                "n": len(qs),
                "mean_target_delta": round(m, 4),
                "null_min_mean": round(null_grp_p, 4) if null_grp_p else None,
                "criterion_met": bool(null_grp_p and m < null_grp_p),
                "pct_beats": round(sum(1 for q in qs if q["beats_null_p5"])/len(qs), 3),
            }

        results["__summary__"] = {
            "target_set":       TARGET_SET,
            "n_null_sets":      len(null_sets),
            "null_seeds":       NULL_SEEDS[:len(null_sets)],
            "null_percentile_idx": NULL_PERCENTILE,
            "n_questions":      len(results) - 1,
            "mean_target_delta": round(mean_target, 4),
            "null_mean_p5":     round(null_mean_p5, 4) if null_mean_p5 else None,
            "pct_beats_null_p5": round(pct_beats, 4),
            "criterion_met":    bool(null_mean_p5 and mean_target < null_mean_p5),
            "by_group": {
                "late_circuit":    group_summary("late_circuit"),
                "embedding_bias":  group_summary("embedding_bias"),
            },
        }
        safe_save(results, out_file)

    print(f"\n[expE'] Done. {len(results)-1} questions + summary saved to {out_file}")
    return results
