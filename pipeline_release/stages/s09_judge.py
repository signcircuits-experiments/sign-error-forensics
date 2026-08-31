"""
s09 — Judge tests on the check layer:
  error domains:   arm B (boost check-layer output on error runs; fixes predicted)
                   + cross-domain boost (same site, other domains)
  correct domains: skipped here — arm A (break corrects) runs from the paired
                   error domain because it needs both datasets.
"""
META = {
    "num": "s09", "name": "judge", "gpu": True,
    "desc": "Boost/weaken the check layer (armA/armB + cross-domain)",
    "out": "exp_judge_candidates_armA/B.json / exp_judge_crossdomain.json",
}


def _candidates(t):
    c = {}
    if t.get("check_layer") is not None:
        c[f"L{t['check_layer']}_mlp"] = {"layer": t["check_layer"], "type": "mlp"}
    c[f"L{t['control_layers'][0]}_mlp"] = {"layer": t["control_layers"][0],
                                           "type": "mlp", "control": True}
    return c


def run(model, domain, args, adapter):
    import experiments.exp_judge_candidates as mJ
    import experiments.exp_judge_crossdomain as mX
    from config import result_path, DOMAIN_PAIRS
    from stages._common import load_df, subset, run_resumable, load_targets, load_result

    t = load_targets(model, domain)
    cands = _candidates(t)
    mJ.CANDIDATES = cands
    mX.CANDIDATES = {k: {"layer": v["layer"], **({"control": True} if v.get("control") else {})}
                     for k, v in cands.items()}
    print(f"[s09] candidates: {list(cands.keys())}")

    if domain in DOMAIN_PAIRS:      # error domain
        df_err = subset(load_df(model, domain), args)
        expA_err = load_result(model, domain, "expA_logit_lens.json")

        outB = result_path(model, domain, "exp_judge_candidates_armB.json",
                           version="v2_mean_ablation")
        run_resumable(
            lambda df, out_file: mJ.run_arm_B(adapter, df, expA_err, out_file),
            df_err, outB, args)

        # arm A: break the paired corrects using error-set means
        cor = DOMAIN_PAIRS[domain]
        df_cor = subset(load_df(model, cor), args)
        outA = result_path(model, cor, "exp_judge_candidates_armA.json",
                           version="v2_mean_ablation")
        run_resumable(
            lambda df, out_file: mJ.run_arm_A(adapter, df,
                                              load_df(model, domain),  # full error set as donors
                                              expA_err, out_file),
            df_cor, outA, args)
        return outB

    # non-paired domain (e.g. another error set): cross-domain boost only
    df = subset(load_df(model, domain), args)
    expA = load_result(model, domain, "expA_logit_lens.json")
    outX = result_path(model, domain, "exp_judge_crossdomain.json",
                       version="v2_mean_ablation")
    run_resumable(
        lambda df, out_file: mX.run_exp_judge_crossdomain(adapter, df, domain,
                                                          out_file, expA),
        df, outX, args)
    return outX
