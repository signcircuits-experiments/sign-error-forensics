"""s04 — Capture projections: h·d̂ of target-layer MLP outputs at the sign position."""
META = {
    "num": "s04", "name": "projections", "gpu": True,
    "desc": "Signed projection of habit/check MLP outputs onto the minus direction, per case",
    "out": "exp_capture_projections.json",
}


def run(model, domain, args, adapter):
    import experiments.exp_capture_projections as m
    from config import results_root
    from stages._common import load_targets
    import os

    t = load_targets(model, domain)
    m.CAPTURE_LAYERS = t["capture_layers"]
    m.DOMAINS = [domain]
    print(f"[s04] capture_layers={m.CAPTURE_LAYERS} domain={domain}")

    out = os.path.join(results_root(model), f"exp_capture_projections_{domain}.json")
    m.run_exp_capture_projections(adapter, model, out)
    return out
