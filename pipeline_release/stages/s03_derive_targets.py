"""
s03 — Derive per-model/per-domain target layers from s01+s02 results.

Writes targets.json:
  habit_layers   : late layers whose MLP most strongly pushes the written '-'
  check_layer    : late layer whose MLP most strongly OPPOSES the written sign
  control_layers : two mid-network boring layers (37.5% and 62.5% depth)
  capture_layers : habit + check (projection stage)
  emb_peak_ids / late_peak_ids, evidence tables, sign token ids

The file is written with human_reviewed=false. REVIEW AND EDIT IT before
running causal stages (s05-s09); set human_reviewed=true when satisfied.
No GPU needed.
"""
import json
import os
from collections import Counter

META = {
    "num": "s03", "name": "derive_targets", "gpu": False,
    "desc": "Derive habit/check/control layers from expA+expB -> targets.json (review by hand!)",
    "out": "targets.json",
}

EMB_PEAK_MAX_LAYER = 5      # peak_layer <= 5 -> embedding-peak case
LATE_WINDOW_FRAC   = 0.75   # habit/check searched in last 25% of layers


def run(model, domain, args, adapter=None):
    from config import MODEL_CONFIGS, targets_path
    from stages._common import load_result

    n_layers = MODEL_CONFIGS[model]["n_layers"]
    late_lo = int(n_layers * LATE_WINDOW_FRAC)

    expA = load_result(model, domain, "expA_logit_lens.json")
    expB = load_result(model, domain, "expB_dla.json")

    # ── Peak split ────────────────────────────────────────────────────────────
    emb_ids, late_ids = [], []
    for qid, q in expA.items():
        pk = q.get("peak_layer")
        (emb_ids if (pk is not None and pk <= EMB_PEAK_MAX_LAYER) else late_ids).append(qid)
    peak_hist = Counter(q.get("peak_layer") for q in expA.values())

    # ── Mean MLP DLA per layer, overall and by written sign ──────────────────
    def mean_mlp(ids):
        acc, cnt = {}, {}
        for qid in ids:
            q = expB.get(qid)
            if not q:
                continue
            for L, v in q["mlp_dla"].items():
                L = int(L)
                acc[L] = acc.get(L, 0.0) + v
                cnt[L] = cnt.get(L, 0) + 1
        return {L: acc[L] / cnt[L] for L in acc}

    all_ids   = list(expB.keys())
    minus_ids = [q for q in all_ids if expB[q]["written_sign"] == "-"]
    plus_ids  = [q for q in all_ids if expB[q]["written_sign"] == "+"]

    mean_all   = mean_mlp(all_ids)
    mean_minus = mean_mlp(minus_ids)
    mean_plus  = mean_mlp(plus_ids)

    late = {L: v for L, v in mean_all.items() if L >= late_lo}

    # habit: in DLA-toward-written units, habit layers push POSITIVE for
    # minus-writers (they push '-', which is what was written).
    late_minus = {L: v for L, v in mean_minus.items() if L >= late_lo} or late
    habit_layers = [L for L, _ in sorted(late_minus.items(),
                                         key=lambda x: -x[1])[:2]]
    habit_layers.sort()

    # check: most NEGATIVE mean DLA toward written sign (opposes the answer)
    check_layer = min(late.items(), key=lambda x: x[1])[0] if late else None

    control_layers = [round(n_layers * 0.375), round(n_layers * 0.625)]

    # sign token ids (mode over cases)
    tok_pairs = Counter((q["wrong_sign_tok"], q["correct_sign_tok"], q["written_sign"])
                        for q in expB.values())

    targets = {
        "model": model, "domain": domain, "n_layers": n_layers,
        "human_reviewed": False,
        "habit_layers": habit_layers,
        "check_layer": check_layer,
        "control_layers": control_layers,
        "capture_layers": sorted(set(habit_layers + ([check_layer] if check_layer is not None else []))),
        "late_window_start": late_lo,
        "n_cases": {"total": len(expA), "emb_peak": len(emb_ids),
                    "late_peak": len(late_ids),
                    "minus_writers": len(minus_ids), "plus_writers": len(plus_ids)},
        "emb_peak_ids": emb_ids,
        "peak_layer_histogram": {str(k): v for k, v in sorted(
            peak_hist.items(), key=lambda x: (x[0] is None, x[0]))},
        "sign_token_modes": [
            {"wrong_sign_tok": a, "correct_sign_tok": b, "written_sign": s, "n": n}
            for (a, b, s), n in tok_pairs.most_common(4)],
        "evidence_mean_mlp_dla": {
            "late_all":   {str(L): round(v, 4) for L, v in sorted(late.items())},
            "late_minus": {str(L): round(v, 4) for L, v in sorted(late_minus.items())},
            "late_plus":  {str(L): round(v, 4) for L, v in sorted(
                {L: v for L, v in mean_plus.items() if L >= late_lo}.items())},
        },
    }

    p = targets_path(model, domain)
    if os.path.exists(p) and not getattr(args, "force", False):
        print(f"[s03] {p} already exists — NOT overwriting (use --force). "
              f"Derived (not saved): habit={habit_layers} check={check_layer}")
        return p
    with open(p, "w") as f:
        json.dump(targets, f, indent=2)
    print(f"[s03] wrote {p}")
    print(f"[s03] habit_layers={habit_layers}  check_layer={check_layer}  "
          f"controls={control_layers}")
    print(f"[s03] REVIEW this file, set human_reviewed=true, then run s04+.")
    return p
