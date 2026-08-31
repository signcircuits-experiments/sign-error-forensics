"""s02 — ExpB direct logit attribution (+ per-head DLA; --matched adds matched-DLA)."""
META = {
    "num": "s02", "name": "dla", "gpu": True,
    "desc": "ExpB DLA — per-layer MLP/attn (+per-head) push toward the written sign",
    "out": "expB_dla.json",
}


def _default_head_layers(n_layers):
    # Late window: last 25% of layers (Qwen 80 -> 60..79 is too many; use last 13)
    return list(range(max(0, n_layers - 13), n_layers))


def run(model, domain, args, adapter):
    from experiments.expB_dla import run_expB
    from config import result_path, MODEL_CONFIGS
    from stages._common import load_df, subset, run_resumable

    df = subset(load_df(model, domain), args)

    if getattr(args, "head_layers", None):
        per_head = [int(x) for x in args.head_layers.split(",")]
    else:
        per_head = _default_head_layers(MODEL_CONFIGS[model]["n_layers"])
    print(f"[s02] per-head DLA layers: {per_head}")

    out = result_path(model, domain, "expB_dla.json")
    run_resumable(
        lambda df, out_file: run_expB(adapter, df, domain, out_file,
                                      per_head_layers=per_head),
        df, out, args)

    if getattr(args, "matched", False):
        from experiments.exp_matched_dla import run_exp_matched_dla
        out_m = result_path(model, domain, "exp_matched_dla.json",
                            version="v2_mean_ablation")
        run_resumable(
            lambda df, out_file: run_exp_matched_dla(adapter, df, domain, out_file),
            df, out_m, args)
    return out
