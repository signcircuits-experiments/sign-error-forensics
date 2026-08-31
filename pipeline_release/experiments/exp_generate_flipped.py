"""
exp_generate_flipped.py — Generation confirmation (S2, corrected rerun)
=========================================================================
PRE-REGISTERED: at the alpha where logit_diff flipped in exp_combined_ablation,
the model's FREELY GENERATED answer's sign is correct for >=1 question.
Baseline (alpha=0) reproduces the error sign.

An audit found three real failure causes in the earlier run (NOT a truncation bug):
  (1) Model declines to compute under greedy decoding for some prompts.
  (2) MAX_NEW_TOKENS=1024 too low — 4x4 det needs 2000–4000 tokens.
  (3) IBP silently dropped: prefix_len cuts prompt mid-marker, so
      marker-in-prompt check failed for all IBP questions.

Four fixes in this version:
  1. MAX_NEW_TOKENS = 4096.
  2. Prompt built via marker index (not prefix_len) — immune to prefix_len errors.
     Assert that prompt endpoint < sign_char_offset.
  3. Record which_pattern_matched + match offset; store HEAD (first 250 chars)
     + TAIL (last 500 chars) since the final answer lives at the end.
     Only count corrected answers from \\boxed or explicit final-answer patterns.
  4. Log declined=True when no mathematical computation is detected — declined
     baseline gives no evidence either way, must not count against prediction.

Alphas: {0 (baseline), 2.0, 3.0}. Both L75.mlp and L78.mlp hooked (C2 stack).
d_hat moved to output.device inside hook body (B1 fix, sharding-safe).

Output: {RESULTS_DIR}/{model}/{domain}/v2_mean_ablation/exp_generate_flipped.json
"""

import os, gc, json, re, shutil, tempfile
import torch
import pandas as pd
from adapters.base_adapter import BaseAdapter
from config import RESULTS_DIR

ALPHAS         = [0, 0.5, 1.0, 2.0, 3.0]

# ── Pipeline parameterization (set by stage from targets.json) ──────────
HL1, HL2 = 75, 78     # habit layers (Qwen defaults; overridden per model)
CTRL1, CTRL2 = 30, 50 # boring control layers
MAX_NEW_TOKENS = 4096   # Fix 1: was 1024; 4x4 det responses are 2000-4000 tokens

# High-confidence final-answer patterns (used for "corrected" claim)
FINAL_PATTERNS = [
    re.compile(r'\\boxed\{\s*([+-])\s*(\d+)'),
    re.compile(r'\\boxed\{\s*([+-]?\d+)'),          # no explicit sign = positive
    re.compile(r'(?:answer|result|value)\s+is\s*([+-])\s*(\d+)', re.IGNORECASE),
    re.compile(r'=\s*\\boxed\{\s*([+-]?\d+)'),
    # IBP fix: sign before expression (e.g. \boxed{-\frac{x^8\ln x}{8}...})
    re.compile(r'\\boxed\{\s*([+-])\s*[\\a-zA-Z\d(]'),
]

# Broader patterns for any sign extraction
BROAD_PATTERNS = [
    re.compile(r'\\boxed\{\s*([+-])\s*\d'),
    re.compile(r'=\s*([+-])\s*\d'),
    re.compile(r'answer(?:\s+is)?:?\s*([+-])\s*\d', re.IGNORECASE),
    re.compile(r'([+-])\s*\d+\s*$'),
]

# Decline indicators: model does not compute, just acknowledges
DECLINE_MARKERS = [
    "please let me know",
    "happy to help",
    "if you have a specific",
    "what would you like",
    "let me know what you need",
]


def _compute_d_hat(adapter) -> torch.Tensor:
    tok   = adapter.tokenizer
    m_ids = tok.encode(' -', add_special_tokens=False)
    p_ids = tok.encode(' +', add_special_tokens=False)
    W_U   = adapter.model.lm_head.weight.detach().float().cpu()
    norm_w = adapter.model.model.norm.weight.detach().float().cpu()
    d = (W_U[m_ids[-1]] - W_U[p_ids[-1]]) * norm_w
    return (d / d.norm()).cpu()


def _mlp(adapter, L):
    return adapter.model.model.layers[L].mlp


def _extract_sign_with_info(text: str) -> dict:
    """
    Extract the final answer sign with pattern metadata.
    Returns dict with: sign, pattern, offset, confidence ('high'|'low'|'none').
    Searches FULL text, takes last (highest-offset) match.
    Fix 3: returns pattern name and offset so declined/uncertain cases are explicit.
    """
    # High-confidence: boxed or explicit "answer is" patterns
    high_matches = []
    for i, pat in enumerate(FINAL_PATTERNS):
        for m in re.finditer(pat, text):
            raw = m.group(1)
            # Extract sign from matched group
            if raw.startswith('-'):
                sign = '-'
            elif raw.startswith('+'):
                sign = '+'
            else:
                sign = '+'  # bare number = positive
            high_matches.append((m.start(), sign, f"FINAL_{i}", m.start()))

    if high_matches:
        _, sign, pat_name, offset = max(high_matches, key=lambda x: x[0])
        return {"sign": sign, "pattern": pat_name, "offset": offset, "confidence": "high"}

    # Broad fallback
    broad_matches = []
    for i, pat in enumerate(BROAD_PATTERNS):
        for m in re.finditer(pat, text):
            sign = m.group(1)
            if sign in ('+', '-'):
                broad_matches.append((m.start(1), sign, f"BROAD_{i}"))

    if broad_matches:
        offset, sign, pat_name = max(broad_matches, key=lambda x: x[0])
        return {"sign": sign, "pattern": pat_name, "offset": offset, "confidence": "low"}

    return {"sign": "?", "pattern": "NONE", "offset": -1, "confidence": "none"}


def _is_declined(text: str) -> bool:
    """Fix 4: detect model decline (no computation, just acknowledgement)."""
    lower = text.lower()
    if any(marker in lower for marker in DECLINE_MARKERS):
        return True
    # Short response with no math content
    if len(text) < 400 and not any(x in text for x in ['\\det', '=', '\\frac', '\\cdot', 'coffactor', 'cofactor', '\\begin{bmatrix}']):
        return True
    return False


def _build_prompt(full_input: str, sign_char_offset: int,
                  marker: str, prefix_len: int) -> tuple:
    """
    Fix 2: build prompt via marker index, not prefix_len.
    Returns (prompt_text, end_idx) or raises ValueError.
    """
    if not marker:
        raise ValueError("response_marker is empty — cannot locate prompt end")

    # Find the marker using index (immune to prefix_len corruption)
    idx = full_input.find(marker)
    if idx == -1:
        raise ValueError(f"response_marker {marker!r} not found in full_input")

    end = idx + len(marker)
    assert end < sign_char_offset, (
        f"Prompt endpoint ({end}) >= sign_char_offset ({sign_char_offset}) — "
        f"sign is inside the prompt, not the response"
    )
    return full_input[:end], end


def _get_flipped_ids(results_dir: str, domain: str) -> dict:
    """Read C2 flipped question IDs + their flip alphas."""
    path = os.path.join(results_dir, domain, "v2_mean_ablation", "exp_combined_ablation.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"exp_combined_ablation not found: {path}")
    with open(path) as f:
        d = json.load(f)
    data = d.get("results", d)

    seen = {}
    for qid, q in data.items():
        for alpha_key in ["a2.0", "a3.0"]:
            e = q.get("conditions", {}).get("C2", {}).get(alpha_key, {})
            if e.get("flipped"):
                alpha = float(alpha_key[1:])
                if qid not in seen or alpha < seen[qid]:
                    seen[qid] = alpha
                break

    print(f"  [genFlipped] {domain}: {len(seen)} C2-flipped questions")
    return seen   # {qid: flip_alpha}


def run_exp_generate_flipped(adapter: BaseAdapter, df: pd.DataFrame,
                              domain: str, out_file: str, results_dir: str):
    cfg    = adapter.cfg
    marker = cfg.get("response_marker", "")

    df_flip = df.copy()
    flipped = {}
    d_hat   = _compute_d_hat(adapter)

    print(f"\n[genFlipped] {domain} | {len(df_flip)} questions | MAX_NEW_TOKENS={MAX_NEW_TOKENS}")

    results = {}

    for _, row in df_flip.iterrows():
        qid              = str(row["id"])
        wrong_sign       = str(row["wrong_sign"])
        correct_sign     = str(row["correct_sign"])
        sign_char_offset = int(row["sign_char_offset"])
        prefix_len       = int(row["prefix_len"])
        full_input       = str(row["full_input"])

        if sign_char_offset <= prefix_len: continue
        if full_input[sign_char_offset] != wrong_sign: continue

        # Fix 2: prompt via marker index (not prefix_len)
        try:
            prompt_text, prompt_end = _build_prompt(
                full_input, sign_char_offset, marker, prefix_len)
        except (ValueError, AssertionError) as e:
            print(f"  ⚠ {qid}: prompt build failed — {e}")
            continue

        prompt_ids = adapter.tokenize(prompt_text)

        q_result = {
            "id":              qid,
            "written_sign":    wrong_sign,
            "correct_sign":    correct_sign,
            "flip_alpha":      flipped.get(qid),
            "prompt_end_char": prompt_end,
            "generations":     {}
        }

        def make_gen_hook(d_hat_cpu, alpha):
            def hook(module, inp, output):
                output = output.clone()
                if output.dim() == 3:
                    d_dev = d_hat_cpu.to(output.device)
                    h     = output[0, -1, :].float()
                    proj  = torch.dot(h, d_dev.float())
                    output[0, -1, :] -= (alpha * proj * d_dev).to(output.dtype)
                return output
            return hook

        for alpha in ALPHAS:

            handles = []
            if alpha > 0:
                handles = [
                    _mlp(adapter, HL1).register_forward_hook(make_gen_hook(d_hat, alpha)),
                    _mlp(adapter, HL2).register_forward_hook(make_gen_hook(d_hat, alpha)),
                ]
            try:
                with torch.no_grad():
                    out_ids = adapter.model.generate(
                        prompt_ids,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        temperature=1.0,
                        pad_token_id=adapter.tokenizer.eos_token_id,
                    )
            finally:
                for h in handles: h.remove()

            new_ids  = out_ids[0, prompt_ids.shape[1]:]
            gen_text = adapter.tokenizer.decode(new_ids, skip_special_tokens=True)

            # Fix 3: extract from FULL text, store head+tail
            sign_info = _extract_sign_with_info(gen_text)
            # Fix 4: detect decline — but high-confidence extraction overrides.
            # A complete answer ending "…please let me know" must not be voided.
            has_decline_marker = _is_declined(gen_text)
            declined = has_decline_marker and sign_info["confidence"] != "high"

            gen_sign = sign_info["sign"]
            correct  = (gen_sign == correct_sign) and not declined
            repro    = (gen_sign == wrong_sign) and not declined

            q_result["generations"][f"alpha_{alpha}"] = {
                "generated_text_head": gen_text[:250],
                "generated_text_tail": gen_text[-500:] if len(gen_text) > 500 else gen_text,
                "generated_text_full": gen_text,
                "generated_length":    len(gen_text),
                "extracted_sign":      gen_sign,
                "which_pattern":       sign_info["pattern"],
                "match_offset":        sign_info["offset"],
                "confidence":          sign_info["confidence"],
                "declined":            declined,
                "correct":             correct,
                "reproduces_error":    repro,
            }
            label = ("DECLINED" if declined else
                     "CORRECT ✓" if correct else
                     "ERROR ✗" if repro else "?")
            print(f"  [{qid}] α={alpha}: {label} sign={gen_sign!r} "
                  f"len={len(gen_text)} pattern={sign_info['pattern']}")
            
            del out_ids, new_ids

        results[qid] = q_result
        del prompt_ids
        gc.collect(); torch.cuda.empty_cache()

        meta = {
            "n_questions":  len(results),
            "alphas":       ALPHAS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "hook_position": "last (-1) at every step (prefill + decoding)",
            "d_hat_recipe":  "space-prefixed, norm-weighted",
            "prompt_method": "marker index (not prefix_len)",
            "prediction":    ("at flip_alpha, generated sign = correct for >=1 question; "
                              "alpha=0 reproduces error sign; declined baseline = no evidence"),
            "pod6_failure_causes": (
                "declines, MAX_NEW_TOKENS=1024 too low, IBP prefix_len cut mid-marker"
            ),
        }
        output = {"meta": meta, "results": results}
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=os.path.dirname(out_file),
                                         suffix=".tmp", delete=False) as f:
            json.dump(output, f, indent=2); tmp = f.name
        shutil.move(tmp, out_file)
        print(f"  [genFlipped] Auto-saved {len(results)} questions to {out_file}")

    print(f"\n[genFlipped] All {len(results)} questions finished → {out_file}")
    return results
