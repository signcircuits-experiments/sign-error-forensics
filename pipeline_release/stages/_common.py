"""
stages/_common.py
=================
Shared helpers for all pipeline stages:
  * experiment_ready loading + schema validation
  * question subsetting (--ids / --range / --limit)
  * per-question resume (skip ids already present in the output JSON)
  * targets.json loading
"""

import os
import json
import shutil
import tempfile
import pandas as pd

from config import data_file, targets_path

REQUIRED_COLS = ["id", "wrong_sign", "correct_sign", "sign_char_offset",
                 "prefix_len", "full_input"]


# ── Data loading ──────────────────────────────────────────────────────────────
def load_df(model: str, domain: str) -> pd.DataFrame:
    path = data_file(model, domain)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"experiment_ready file not found: {path}\n"
            f"Convention: {{DATA_DIR}}/{model}_{domain}_experiment_ready.xlsx "
            f"(override SIGN_DATA_DIR if data lives elsewhere)")
    df = pd.read_excel(path) if path.endswith(".xlsx") else pd.read_csv(path)
    if "id" not in df.columns and "Problem_ID" in df.columns:
        df = df.rename(columns={"Problem_ID": "id"})
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{os.path.basename(path)} missing columns: {missing}")
    df["id"] = df["id"].astype(str)
    print(f"[data] {model}/{domain}: {len(df)} rows from {os.path.basename(path)}")
    return df


# ── Subsetting ────────────────────────────────────────────────────────────────
def subset(df: pd.DataFrame, args) -> pd.DataFrame:
    """Apply --ids / --range / --limit (in that priority order)."""
    if getattr(args, "ids", None):
        want = [s.strip() for s in args.ids.split(",")]
        df = df[df["id"].isin(want)]
        print(f"[subset] --ids -> {len(df)} rows")
    elif getattr(args, "range", None):
        a, b = args.range.split("-")
        df = df.iloc[int(a) - 1: int(b)]          # 1-based inclusive
        print(f"[subset] --range {args.range} -> {len(df)} rows")
    if getattr(args, "limit", None):
        df = df.head(int(args.limit))
        print(f"[subset] --limit -> {len(df)} rows")
    return df


# ── Resume ────────────────────────────────────────────────────────────────────
def _load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _safe_save(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".tmp", delete=False) as f:
        json.dump(data, f, indent=2)
        tmp = f.name
    shutil.move(tmp, path)


def run_resumable(fn, df: pd.DataFrame, out_file: str, args, **kw):
    """
    Run experiment function fn(df=..., out_file=...) with per-question resume.

    Works for experiments whose output JSON is a dict keyed by question id
    (all sNN observation/causal stages follow this convention).

    Mechanics: already-done ids are dropped from df; fn writes to a .part file;
    afterwards part + existing are merged into out_file (existing rows kept).
    """
    part = out_file + ".part.json"
    no_resume = getattr(args, "no_resume", False)

    existing = _load_json(out_file)
    if not isinstance(existing, dict):
        existing = {}

    # Salvage rows from a .part file left behind by a crashed/interrupted run.
    leftover = _load_json(part)
    if isinstance(leftover, dict) and leftover and not no_resume:
        existing = {**existing, **leftover}
        _safe_save(existing, out_file)
        print(f"[resume] salvaged {len(leftover)} rows from interrupted run")
    if os.path.exists(part):
        os.remove(part)

    if no_resume:
        existing = {}
    elif existing:
        done = set(existing.keys())
        before = len(df)
        df = df[~df["id"].isin(done)]
        print(f"[resume] {len(done)} ids already in {os.path.basename(out_file)}; "
              f"{before - len(df)} skipped, {len(df)} to run")
        if df.empty:
            print("[resume] nothing to do")
            return existing

    fn(df=df, out_file=part, **kw)

    new = _load_json(part) or {}
    merged = {**existing, **new}
    _safe_save(merged, out_file)
    if os.path.exists(part):
        os.remove(part)
    print(f"[resume] merged {len(new)} new + {len(existing)} existing "
          f"-> {len(merged)} in {out_file}")
    return merged


# ── Targets ───────────────────────────────────────────────────────────────────
def load_targets(model: str, domain: str) -> dict:
    """Load targets.json (written by s03, human-reviewed). Hard error if absent."""
    p = targets_path(model, domain)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"targets.json not found: {p}\n"
            f"Run stage s03 (derive targets) first, review the file, then rerun.")
    with open(p) as f:
        t = json.load(f)
    if not t.get("human_reviewed", False):
        print(f"[targets] WARNING: {p} has human_reviewed=false. "
              f"Review the derived layers before trusting causal-stage results.")
    return t


def load_result(model: str, domain: str, fname: str, version: str = None) -> dict:
    from config import result_path
    p = result_path(model, domain, fname, version)
    data = _load_json(p)
    if data is None:
        raise FileNotFoundError(f"Required result not found: {p}")
    return data
