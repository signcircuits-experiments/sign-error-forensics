"""
s05 — Mean-ablations (correlational): expD' single-site MLP/attn, expE' joint,
optionally expC' per-head (--with-heads path/to/heads.json).

Target sites come from targets.json: habit + check + top |DLA| late layers
+ boring controls. One site per run, never combinations (expE' handles joints).
"""
META = {
    "num": "s05", "name": "ablations", "gpu": True,
    "desc": "expD' single-site mean-ablation + expE' joint ablation (+expC' heads, optional)",
    "out": "expD_prime_mlp_mean_ablation.json / expE_prime_joint_mean_ablation.json",
}

N_TOP_SITES = 5   # extra late layers by |mean DLA| beyond habit/check


def _build_mlp_targets(t):
    """Construct expD'-style MLP_TARGETS dict from targets.json."""
    sites = {}
    late = {int(L): v for L, v in t["evidence_mean_mlp_dla"]["late_all"].items()}
    for L in t["habit_layers"]:
        sites[f"L{L}_habit"] = {"layer": L, "role": "pusher",
                                "prior_DLA_pct": 0.0, "type": "mlp"}
    if t.get("check_layer") is not None:
        L = t["check_layer"]
        sites[f"L{L}_check"] = {"layer": L, "role": "corrector",
                                "prior_DLA_pct": 0.0, "type": "mlp"}
    used = {v["layer"] for v in sites.values()}
    extra = sorted((L for L in late if L not in used),
                   key=lambda L: -abs(late[L]))[:N_TOP_SITES]
    for L in extra:
        role = "pusher" if late[L] > 0 else "corrector"
        sites[f"L{L}_{role[:4]}"] = {"layer": L, "role": role,
                                     "prior_DLA_pct": 0.0, "type": "mlp"}
    for L in t["control_layers"]:
        sites[f"L{L}_ctrl"] = {"layer": L, "role": "control",
                               "prior_DLA_pct": 0.0, "type": "mlp"}
    return sites


def run(model, domain, args, adapter):
    import experiments.expD_prime as mD
    import experiments.expE_prime as mE
    from config import result_path
    from stages._common import load_df, subset, run_resumable, load_targets, load_result

    t = load_targets(model, domain)
    df = subset(load_df(model, domain), args)
    expA = load_result(model, domain, "expA_logit_lens.json")

    mD.MLP_TARGETS = _build_mlp_targets(t)
    print(f"[s05] expD' sites: {list(mD.MLP_TARGETS.keys())}")
    outD = result_path(model, domain, "expD_prime_mlp_mean_ablation.json",
                       version="v2_mean_ablation")
    run_resumable(
        lambda df, out_file: mD.run_expD_prime(adapter, df, domain, out_file, expA),
        df, outD, args)

    # expE' joint target set from targets.json (habit + top late |DLA| layers, 5 total)
    late = {int(L): v for L, v in t["evidence_mean_mlp_dla"]["late_all"].items()}
    joint = list(t["habit_layers"])
    for L in sorted((L for L in late if L not in joint), key=lambda L: -abs(late[L])):
        if len(joint) >= 5:
            break
        joint.append(L)
    mE.TARGET_SET = sorted(joint)
    print(f"[s05] expE' TARGET_SET: {mE.TARGET_SET}")

    outE = result_path(model, domain, "expE_prime_joint_mean_ablation.json",
                       version="v2_mean_ablation")
    run_resumable(
        lambda df, out_file: mE.run_expE_prime(adapter, df, domain, out_file, expA),
        df, outE, args)

    if getattr(args, "with_heads", None):
        from experiments.expC_prime import run_expC_prime
        outC = result_path(model, domain, "expC_prime_head_mean_ablation.json",
                           version="v2_mean_ablation")
        run_resumable(
            lambda df, out_file: run_expC_prime(adapter, df, domain, out_file,
                                                expA, args.with_heads),
            df, outC, args)
    return outD
