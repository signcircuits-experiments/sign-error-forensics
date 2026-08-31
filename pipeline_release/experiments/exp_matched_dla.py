"""
exp_matched_dla.py — Text-controlled base-vs-instruct DLA comparison (S6c)
============================================================================
Runs expB-style DLA on the SAME base-generated full_inputs through BOTH models.
Base model and Instruct model see identical text — model difference, not text
difference.

Method:
  Load base-model experiment-ready xlsx (built by build_base_experiment_ready.py).
  Run expB DLA on those full_inputs through:
    (A) qwen_base   → {RESULTS_DIR}/qwen_base/arith_simple/v2_mean_ablation/expB_dla.json
    (B) qwen        → {RESULTS_DIR}/qwen/arith_simple_base_text/v2_mean_ablation/expB_dla.json
  Compare L78 MLP DLA split by written sign between A and B.

PRE-REGISTERED:
  L78 '−' push present in base model (A) → pretraining origin.
  Absent → instruct-tuning artifact. Either result is publishable.

Usage:
    python run_pod6_experiments.py --job matched_dla
    (Swaps model — see runner. Can also run this file directly if models are loaded
    in sequence externally.)

Note: runner swaps model between A and B; this script is called TWICE with
different adapters and output paths.
"""

import os, gc, json
import pandas as pd
from adapters.base_adapter import BaseAdapter
from config import RESULTS_DIR
from experiments.expB_dla import run_expB


def run_exp_matched_dla(adapter: BaseAdapter, df: pd.DataFrame,
                         domain_tag: str, out_file: str):
    """
    Run expB DLA on df (base-generated text) through the currently loaded adapter.
    domain_tag: 'arith_simple' (base) or 'arith_simple_base_text' (instruct on base text).
    """
    per_head = [67, 71, 72, 74, 75, 76, 77, 78, 79]
    print(f"\n[expMatchedDLA] {adapter.model_name} | domain_tag={domain_tag} | n={len(df)}")
    run_expB(adapter, df, domain_tag, out_file, per_head_layers=per_head)
    print(f"[expMatchedDLA] → {out_file}")
