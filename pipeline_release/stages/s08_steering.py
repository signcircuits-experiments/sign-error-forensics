"""
s08 — C2 steering: subtract the minus direction at BOTH habit layers during the
forward pass (error domains) or on correct runs (correct domains), plus the
magnitude-matched random-direction control and real generation of flipped answers.
"""
META = {
    "num": "s08", "name": "steering", "gpu": True,
    "desc": "C2 steering at habit layers + ctrl_mag control + generate flipped answers",
    "out": "exp_combined_ablation.json / exp_c2_on_correct.json / exp_ctrl_mag.json / exp_generate_flipped.json",
}


def _set_layers(mod, t):
    hl = t["habit_layers"]
    if len(hl) != 2:
        raise ValueError(
            f"s08 needs exactly 2 habit_layers in targets.json (got {hl}). "
            f"Edit targets.json (the C2 arm steers two sites).")
    mod.HL1, mod.HL2 = hl[0], hl[1]
    c = t["control_layers"]
    mod.CTRL1, mod.CTRL2 = c[0], c[-1]
    print(f"[s08] {mod.__name__}: habit=({mod.HL1},{mod.HL2}) "
          f"controls=({mod.CTRL1},{mod.CTRL2})")


def run(model, domain, args, adapter):
    from config import result_path, CORRECT_DATASETS, results_root
    from stages._common import load_df, subset, run_resumable, load_targets, load_result

    t = load_targets(model, domain)
    df = subset(load_df(model, domain), args)
    expA = load_result(model, domain, "expA_logit_lens.json")

    if domain in CORRECT_DATASETS:
        import experiments.exp_c2_on_correct as m
        _set_layers(m, t)
        out = result_path(model, domain, "exp_c2_on_correct.json",
                          version="v2_mean_ablation")
        run_resumable(
            lambda df, out_file: m.run_exp_c2_on_correct(adapter, df, out_file, expA),
            df, out, args)
        return out

    import experiments.exp_combined_ablation as mC
    import experiments.exp_ctrl_mag as mM
    import experiments.exp_generate_flipped as mG
    for mod in (mC, mM, mG):
        _set_layers(mod, t)

    out = result_path(model, domain, "exp_combined_ablation.json",
                      version="v2_mean_ablation")
    run_resumable(
        lambda df, out_file: mC.run_exp_combined_ablation(adapter, df, domain,
                                                          out_file, expA),
        df, out, args)

    outM = result_path(model, domain, "exp_ctrl_mag.json",
                       version="v2_mean_ablation")
    run_resumable(
        lambda df, out_file: mM.run_exp_ctrl_mag(adapter, df, domain, out_file, expA),
        df, outM, args)

    if not getattr(args, "skip_generation", False):
        outG = result_path(model, domain, "exp_generate_flipped.json",
                           version="v2_mean_ablation")
        run_resumable(
            lambda df, out_file: mG.run_exp_generate_flipped(
                adapter, df, domain, out_file, results_root(model)),
            df, outG, args)
    return out
