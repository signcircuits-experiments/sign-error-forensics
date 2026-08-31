"""s01 — ExpA logit lens: sign preference at every layer, per question."""
META = {
    "num": "s01", "name": "logit_lens", "gpu": True,
    "desc": "ExpA logit lens — layer-by-layer sign preference at the sign position",
    "out": "expA_logit_lens.json",
}


def run(model, domain, args, adapter):
    from experiments.expA_logit_lens import run_expA
    from config import result_path
    from stages._common import load_df, subset, run_resumable

    df = subset(load_df(model, domain), args)
    out = result_path(model, domain, "expA_logit_lens.json")
    run_resumable(
        lambda df, out_file: run_expA(adapter, df, domain, out_file),
        df, out, args)
    return out
