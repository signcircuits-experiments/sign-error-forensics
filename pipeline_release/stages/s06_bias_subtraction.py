"""s06 — ExpG' bias subtraction (donor-free causal test) at habit/control sites."""
META = {
    "num": "s06", "name": "bias_subtraction", "gpu": True,
    "desc": "expG' subtract minus-direction projection at habit sites (+boring controls)",
    "out": "expG_prime_bias_subtraction.json",
}


def _build_targets(t):
    sites = {}
    for L in t["habit_layers"]:
        sites[f"L{L}_mlp"] = {"layer": L, "type": "mlp", "role": "fixed_minus"}
    for L in t["control_layers"]:
        sites[f"L{L}_mlp"] = {"layer": L, "type": "mlp", "role": "control_boring"}
    return sites


def run(model, domain, args, adapter):
    import experiments.expG_prime as m
    from config import result_path
    from stages._common import load_df, subset, run_resumable, load_targets, load_result

    t = load_targets(model, domain)
    m.TARGETS = _build_targets(t)
    print(f"[s06] expG' sites: {list(m.TARGETS.keys())}")

    df = subset(load_df(model, domain), args)
    expA = load_result(model, domain, "expA_logit_lens.json")
    out = result_path(model, domain, "expG_prime_bias_subtraction.json",
                      version="v2_mean_ablation")
    run_resumable(
        lambda df, out_file: m.run_expG_prime(adapter, df, domain, out_file, expA),
        df, out, args)
    return out
