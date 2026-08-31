"""
s07 — Activation patching between correct and error runs (error domains only):
  expF'   correct-mean -> error runs (habit sites predicted null; check predicted movement)
  reverse error-mean  -> correct runs (must break '+'-corrects to close causality)
  sign-matched patch  (same-sign donors; controls for sign composition)
Requires the paired correct domain (DOMAIN_PAIRS) and its expA.
"""
META = {
    "num": "s07", "name": "patching", "gpu": True,
    "desc": "expF' / reverse / sign-matched single-site patches between correct and error runs",
    "out": "expF_prime_correct_mean_patch.json etc.",
}


def _site_dicts(t):
    full, small = {}, {}
    for L in t["habit_layers"]:
        full[f"L{L}_mlp"] = {"layer": L, "type": "mlp",
                             "prediction": "null", "role": "fixed_minus"}
        small[f"L{L}_mlp"] = {"layer": L, "role": "fixed_minus", "type": "mlp"}
    if t.get("check_layer") is not None:
        L = t["check_layer"]
        full[f"L{L}_mlp"] = {"layer": L, "type": "mlp",
                             "prediction": "movement", "role": "check"}
    for L in t["control_layers"]:
        full[f"L{L}_mlp"] = {"layer": L, "type": "mlp",
                             "prediction": "null", "role": "control"}
    small[f"L{t['control_layers'][0]}_mlp"] = {
        "layer": t["control_layers"][0], "role": "control", "type": "mlp"}
    return full, small


def run(model, domain, args, adapter):
    import experiments.expF_prime as mF
    import experiments.exp_reverse_patch as mR
    import experiments.exp_signmatched_patch as mS
    from config import result_path, DOMAIN_PAIRS
    from stages._common import load_df, subset, run_resumable, load_targets, load_result

    if domain not in DOMAIN_PAIRS:
        raise ValueError(f"s07 runs on error domains with a correct pair; "
                         f"'{domain}' has no pair in DOMAIN_PAIRS.")
    cor_domain = DOMAIN_PAIRS[domain]

    t = load_targets(model, domain)
    full_sites, small_sites = _site_dicts(t)
    mF.TARGETS = full_sites
    mR.TARGETS = small_sites
    mS.TARGETS = full_sites
    print(f"[s07] sites: {list(full_sites.keys())}  (donor domain: {cor_domain})")

    df_err = subset(load_df(model, domain), args)
    df_cor = load_df(model, cor_domain)          # donors: always full set
    expA_err = load_result(model, domain, "expA_logit_lens.json")

    outF = result_path(model, domain, "expF_prime_correct_mean_patch.json",
                       version="v2_mean_ablation")
    run_resumable(
        lambda df, out_file: mF.run_expF_prime(adapter, df, df_cor, domain,
                                               out_file, expA_err),
        df_err, outF, args)

    outR = result_path(model, domain, "exp_reverse_patch.json",
                       version="v2_mean_ablation")
    # reverse: subset applies to the CORRECT (patched) side
    df_cor_sub = subset(load_df(model, cor_domain), args)
    run_resumable(
        lambda df, out_file: mR.run_exp_reverse_patch(adapter, df, df_err,
                                                      out_file, expA_err),
        df_cor_sub, outR, args)

    outS = result_path(model, domain, "exp_signmatched_patch.json",
                       version="v2_mean_ablation")
    run_resumable(
        lambda df, out_file: mS.run_exp_signmatched_patch(adapter, df, df_cor,
                                                          out_file, expA_err),
        df_err, outS, args)
    return outF
