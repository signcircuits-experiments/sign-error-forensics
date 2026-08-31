"""
derive_s05_release_json.py — raw s05 output → released per-case JSON
=====================================================================
Documents and reproduces the transform that produced the released
`{domain}_habit_ablation.json` and `{domain}_joint_ablation.json` files
(per-case, flat corrector_L*/control_L* format) from the raw pipeline
outputs written by s05_ablations.py:

  habit : expD_prime_mlp_mean_ablation.json   (per-case, nested `ablations`)
  joint : expE_prime_joint_mean_ablation.json (per-case, flat)

`written_sign` and `sign_subtype` are joined in from the s01 output
(expA_logit_lens.json), which carries both fields per case id.

Transform:
  habit : for each case, keep id/domain/baseline_ld/peak_layer, join
          written_sign/sign_subtype from expA, then for each site in
          SITE_MAP emit {prefix}_delta_ld / {prefix}_ablated_ld /
          {prefix}_role from the raw `ablations` dict. All other raw
          detail (repair matrices, recovery curves, per-site DLA) stays
          in the raw file, which remains ground truth.
  joint : raw record verbatim + written_sign + sign_subtype appended.

Usage:
  python tools/derive_s05_release_json.py habit \
      --raw  .../expD_prime_mlp_mean_ablation.json \
      --expa .../expA_logit_lens.json \
      --out  det_4x4_error_habit_ablation.json
  python tools/derive_s05_release_json.py joint \
      --raw  .../expE_prime_joint_mean_ablation.json \
      --expa .../expA_logit_lens.json \
      --verify released/det_4x4_error_joint_ablation.json

  --out    write the derived JSON to this path.
  --verify regenerate and compare against an existing released file;
           exits non-zero on any mismatch (no file is written).
"""

import argparse
import json
import sys

# released field prefix -> raw expD' site-key aliases (discovery and held-out
# runs of s05 used different site names for the same layers)
SITE_MAP = {
    "corrector_L75": ("L75_corr", "L75_habit"),
    "corrector_L78": ("L78_corr", "L78_habit"),
    "corrector_L79": ("L79_corr", "L79_check"),
    "control_L30": ("L30_ctrl",),
    "control_L50": ("L50_ctrl",),
}


def _load(path):
    with open(path) as f:
        d = json.load(f)
    return d.get("results", d)


def _sign_lookup(expa_path):
    expa = _load(expa_path)
    return {
        str(cid): (rec.get("written_sign"), rec.get("sign_subtype"))
        for cid, rec in expa.items()
        if isinstance(rec, dict)
    }


def _is_case(rec):
    return isinstance(rec, dict) and "id" in rec


def derive_habit(raw_path, expa_path):
    raw = _load(raw_path)
    signs = _sign_lookup(expa_path)
    out = {}
    for cid, rec in raw.items():
        if not _is_case(rec):
            continue  # e.g. a '__summary__' block
        ws, st = signs.get(str(cid), (None, None))
        row = {
            "id": rec["id"],
            "domain": rec["domain"],
            "written_sign": ws,
            "sign_subtype": st,
            "baseline_ld": rec["baseline_ld"],
            "peak_layer": rec["peak_layer"],
        }
        for prefix, aliases in SITE_MAP.items():
            site = next(s for s in aliases if s in rec["ablations"])
            a = rec["ablations"][site]
            row[f"{prefix}_delta_ld"] = a["delta_ld"]
            row[f"{prefix}_ablated_ld"] = a["ablated_ld"]
            row[f"{prefix}_role"] = a["role"]
        out[cid] = row
    return out


def derive_joint(raw_path, expa_path):
    raw = _load(raw_path)
    signs = _sign_lookup(expa_path)
    out = {}
    for cid, rec in raw.items():
        if not _is_case(rec):
            continue  # e.g. a '__summary__' block
        ws, st = signs.get(str(cid), (None, None))
        row = dict(rec)
        row["written_sign"] = ws
        row["sign_subtype"] = st
        out[cid] = row
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("kind", choices=["habit", "joint"])
    p.add_argument("--raw", required=True,
                   help="raw expD'/expE' JSON from s05_ablations.py")
    p.add_argument("--expa", required=True,
                   help="expA_logit_lens.json from s01 (sign/subtype source)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--out", help="write derived JSON here")
    g.add_argument("--verify", help="compare regeneration against this "
                                    "released file; write nothing")
    args = p.parse_args()

    derived = (derive_habit if args.kind == "habit"
               else derive_joint)(args.raw, args.expa)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(derived, f, indent=2)
        print(f"[derive_s05] wrote {len(derived)} cases -> {args.out}")
        return

    with open(args.verify) as f:
        released = json.load(f)
    if derived == released:
        print(f"[derive_s05] VERIFIED: {args.verify} "
              f"({len(released)} cases) matches regeneration exactly.")
        return
    missing = set(released) - set(derived)
    extra = set(derived) - set(released)
    diff_cases = [c for c in set(released) & set(derived)
                  if released[c] != derived[c]]
    print(f"[derive_s05] MISMATCH vs {args.verify}: "
          f"missing={sorted(missing)[:5]} extra={sorted(extra)[:5]} "
          f"differing={sorted(diff_cases)[:5]} "
          f"(counts: {len(missing)}/{len(extra)}/{len(diff_cases)})")
    for c in sorted(diff_cases)[:3]:
        for k in released[c]:
            if released[c].get(k) != derived[c].get(k):
                print(f"  {c}.{k}: released={released[c].get(k)!r} "
                      f"derived={derived[c].get(k)!r}")
    sys.exit(1)


if __name__ == "__main__":
    main()
