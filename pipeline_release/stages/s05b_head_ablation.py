"""
s05b — Generalized per-head mean-ablation (EXPLORATORY, new stage)
====================================================================

*** EXPLORATORY — head ablation was NOT in the pre-registered experiment
set for the 3 new small models (gemma-3-27b-it, phi-4,
mistral-small-3.2-24b-instruct). Unlike s05_ablations.py's expD'/expE'
(pre-registered MLP/joint ablation targets derived from targets.json), this
stage requires an explicit --sites JSON file with human-chosen
(layer, head) targets — there is no automatic target derivation here. ***

INTEGRATION CHOICE (documented per Part 3 instructions):
  s05_ablations.py's existing `--with-heads path/to/heads.json` flag calls
  `experiments.expC_prime.run_expC_prime`, which is Llama-specific
  (hardcoded L75/L77 pre-registered targets + Llama-specific self-repair
  layers, see expC_prime.py docstring). Two integration options were
  considered:
    (1) Extend s05_ablations.py's existing --with-heads branch to call the
        new generalized exp_head_ablation.py instead of expC_prime.py.
    (2) Add a NEW stage file (this file, s05b) that wires the generalized
        module independently, leaving s05_ablations.py/--with-heads/
        expC_prime.py untouched for Llama.
  CHOSEN: option (2), a new stage file. Rationale:
    - expC_prime.py's pre-registered Llama target/self-repair design is
      still the correct implementation for the Llama det-domain analysis
      it was built for; overwriting --with-heads's behavior to point at a
      different (site-list-driven, no self-repair-by-default) module would
      silently change what --with-heads does for EXISTING Llama runs/
      results, which is a much bigger blast radius than adding a new,
      clearly-labeled exploratory stage for the 3 new models only.
    - The new models need a --sites JSON file (no pre-registered targets
      exist yet), which is a different CLI contract than --with-heads's
      "path to expB_dla_l75_heads.json" (a DLA-results file, not a raw
      site list) — conflating the two flags' semantics would be confusing.
    - This keeps s05_ablations.py's diff to ZERO (out of caution, since it
      is shared/working code for the existing Llama/Qwen pipeline), and
      keeps the new, unverified, EXPLORATORY code fully isolated in its own
      file/stage number, matching the pattern the pipeline already uses for
      one-experiment-per-stage-file elsewhere (s06..s09).
  This stage IS registered in run.py's STAGE_MODULES list; `python run.py
  --stages 05b` resolves it via --list/--stages parsing. The adapters for
  the 3 new small models are included for forthcoming releases and are
  untested against the current Qwen det_4x4 data drop.

Target sites: supplied via --sites path/to/sites.json (schema documented in
exp_head_ablation.py's module docstring). --dry-run resolves all hooks
without any forward pass (see exp_head_ablation.run_dry_run).
"""

META = {
    "num": "s05b", "name": "head_ablation", "gpu": True,
    "desc": "EXPLORATORY generalized per-head mean-ablation via --sites JSON "
            "(adapter-routed, model-agnostic; NOT pre-registered for the 3 "
            "new small models). Use --dry-run to resolve hooks with no "
            "forward pass before spending GPU time.",
    "out": "exp_head_ablation.json",
}


def run(model, domain, args, adapter):
    import json
    from config import result_path
    from stages._common import load_df, subset, run_resumable
    from experiments.exp_head_ablation import run_head_ablation, run_dry_run

    sites_path = getattr(args, "sites", None)
    if not sites_path:
        raise SystemExit(
            "s05b requires --sites path/to/sites.json "
            "(schema: list of {layer, head, role, name} — see "
            "exp_head_ablation.py docstring). This is EXPLORATORY: there is "
            "no pre-registered target list for the 3 new models.")
    with open(sites_path) as f:
        sites = json.load(f)
    print(f"[s05b] Loaded {len(sites)} sites from {sites_path}")

    dry_run = getattr(args, "dry_run", False)
    if dry_run:
        report = run_dry_run(adapter, sites)
        out = result_path(model, domain, "exp_head_ablation_dry_run_report.json")
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[s05b] Dry-run report written to {out}. "
              f"No forward pass was run; re-run WITHOUT --dry-run once all "
              f"sites report status=OK.")
        return report

    zero_fallback = getattr(args, "zero_ablation_fallback", False)
    self_repair_layers = []
    if getattr(args, "self_repair_layers", None):
        self_repair_layers = [int(x) for x in args.self_repair_layers.split(",")]

    df = subset(load_df(model, domain), args)
    out_file = result_path(model, domain, "exp_head_ablation.json")

    run_resumable(
        lambda df, out_file: run_head_ablation(
            adapter, df, domain, out_file, sites,
            zero_ablation_fallback=zero_fallback,
            self_repair_layers=self_repair_layers),
        df, out_file, args)
    return out_file
