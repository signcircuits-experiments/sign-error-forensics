#!/usr/bin/env python3
"""
run.py — sign-circuit pipeline driver
=====================================
One command, numbered stages, any model.

  python run.py --list
  python run.py --model llama --domain det_4x4_error --stages 01,02 --limit 5
  python run.py --model llama --domain det_4x4_error --stages 01-04
  python run.py --model llama --domain det_4x4_error --stages 03      # local, no GPU
  python run.py --model qwen  --domain det_4x4_error --stages 13,14   # local

Contract: input is one experiment_ready file per (model, domain):
  {SIGN_DATA_DIR}/{model}_{domain}_experiment_ready.xlsx
Results: {SIGN_RESULTS_DIR}/{Model}/{domain}/...

Subsetting (any GPU stage): --ids q1,q2 | --range 1-25 | --limit 5
Resume: already-computed question ids are skipped automatically (--no-resume to redo).
The model is loaded ONCE per invocation and shared across the selected stages.
"""

import argparse
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

STAGE_MODULES = [
    "s00_arith_control",
    "s01_logit_lens",
    "s02_dla",
    "s03_derive_targets",
    "s04_projections",
    "s05_ablations",
    "s05b_head_ablation",
    "s06_bias_subtraction",
    "s07_patching",
    "s08_steering",
    "s09_judge",
]


def load_stages():
    stages = {}
    for name in STAGE_MODULES:
        mod = importlib.import_module(f"stages.{name}")
        stages[mod.META["num"]] = mod
    return stages


def parse_stage_selection(sel, stages):
    """'01,02' or '01-04' or 'all' -> ordered stage nums."""
    nums = sorted(stages.keys())
    if sel in (None, "all"):
        return nums
    picked = []
    for part in sel.split(","):
        part = part.strip()
        # numeric range, e.g. "01-04" (letter-suffixed stages like 05b are
        # never part of a range -- select them individually, e.g. "05,05b")
        if "-" in part and part.replace("-", "").isdigit():
            a, b = part.split("-")
            picked += [n for n in nums if int(a) <= int(n[1:3]) <= int(b)]
        else:
            digits = "".join(c for c in part if c.isdigit())
            suffix = part[len(digits):] if part[:1].isdigit() else part.lstrip("sS")[len(digits):]
            n = f"s{int(digits):02d}{suffix.lower()}" if digits else part.lower()
            if n not in stages:
                raise SystemExit(f"Unknown stage {part}. Use --list.")
            picked.append(n)
    return sorted(set(picked))


def main():
    stages = load_stages()

    p = argparse.ArgumentParser(description="Sign-circuit pipeline")
    p.add_argument("--list", action="store_true", help="show stages and exit")
    p.add_argument("--model", help="qwen | qwen_base | llama | llama_base")
    p.add_argument("--domain", help="e.g. det_4x4_error (see config.KNOWN_DOMAINS)")
    p.add_argument("--stages", help="e.g. 01,02 or 01-04 or all")
    # subsetting / resume
    p.add_argument("--ids", help="comma-separated question ids")
    p.add_argument("--range", help="1-based inclusive row range, e.g. 1-25")
    p.add_argument("--limit", type=int, help="first N questions (smoke test)")
    p.add_argument("--no-resume", action="store_true", dest="no_resume",
                   help="recompute even if ids already in output")
    # stage-specific
    p.add_argument("--head-layers", dest="head_layers",
                   help="s02: comma-separated layers for per-head DLA")
    p.add_argument("--matched", action="store_true", help="s02: also run matched DLA")
    p.add_argument("--force", action="store_true", help="s03: overwrite targets.json")
    p.add_argument("--with-heads", dest="with_heads",
                   help="s05: path to per-head DLA json enabling expC'")
    p.add_argument("--sites", help="s05b: path to sites.json (list of {layer,head,role,name})")
    p.add_argument("--dry-run", action="store_true", dest="dry_run",
                   help="s05b: resolve hooks only, no forward pass")
    p.add_argument("--zero-ablation-fallback", action="store_true", dest="zero_ablation_fallback",
                   help="s05b: zero-ablate instead of mean-ablate")
    p.add_argument("--self-repair-layers", dest="self_repair_layers",
                   help="s05b: comma-separated layer list for optional self-repair logging")
    p.add_argument("--skip-generation", action="store_true", dest="skip_generation",
                   help="s08: skip the (slow) generation arm")
    args = p.parse_args()

    if args.list or not (args.model and args.domain and args.stages):
        print(f"{'num':4s} {'name':18s} {'gpu':3s}  description")
        print("-" * 78)
        for n in sorted(stages):
            m = stages[n].META
            print(f"{m['num']:4s} {m['name']:18s} {'GPU' if m['gpu'] else '-':3s}  {m['desc']}")
        if not args.list:
            print("\nNeed --model, --domain and --stages to run. Example:\n"
                  "  python run.py --model llama --domain det_4x4_error --stages 01,02 --limit 5")
        return

    import config  # noqa: prints version info, defines paths
    from config import MODEL_CONFIGS
    if args.model not in MODEL_CONFIGS:
        raise SystemExit(f"Unknown model '{args.model}'. Options: {list(MODEL_CONFIGS)}")

    picked = parse_stage_selection(args.stages, stages)
    needs_gpu = any(stages[n].META["gpu"] for n in picked)
    print(f"\n=== model={args.model} domain={args.domain} stages={picked} "
          f"gpu={'yes' if needs_gpu else 'no'} ===\n")

    adapter = None
    if needs_gpu:
        from adapters import get_adapter
        adapter = get_adapter(args.model, MODEL_CONFIGS[args.model])
        adapter.load()

    try:
        for n in picked:
            m = stages[n]
            print(f"\n──── {m.META['num']} {m.META['name']} ────")
            m.run(args.model, args.domain, args, adapter)
    finally:
        if adapter is not None:
            print("\n[run] unloading model ...")
            adapter.unload()
    print("\n[run] done.")


if __name__ == "__main__":
    main()
