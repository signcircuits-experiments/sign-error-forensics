"""
s00 — Arithmetic control set builder.

Every model is tested on the SAME 48 arithmetic problems
(data_shared/arith_control_problems.json, extracted from the original Qwen
arith_simple set). This stage generates the model's OWN solutions to those
problems and writes a standard experiment_ready file, so arith_simple can then
flow through s01/s02 like any other domain.

Purpose of the control: the arithmetic DLA profile at the late layers
(L75–79) separates "general sign circuit" from "task-specific circuit" and
supports base-vs-instruct origin comparisons — identical prompts across all
models, so profiles are directly comparable.

Usage:
    python run.py --model llama --domain arith_simple --stages 00
    # then: --stages 01,02 as usual

Output: {DATA_DIR}/{model}_arith_simple_experiment_ready.xlsx
(never overwrites an existing file; delete it manually to rebuild)

Column convention matches all experiment_ready files:
  wrong_sign   = the sign the model WROTE (error-domain heritage naming)
  correct_sign = the other sign
Extra columns: true_sign (ground truth), written_correct.
"""
import os
import re

META = {
    "num": "s00", "name": "arith_control", "gpu": True,
    "desc": "generate model's own solutions to the shared 48-problem arithmetic control set",
    "out": "{model}_arith_simple_experiment_ready.xlsx (in DATA_DIR)",
}

MAX_NEW_TOKENS = 1024


def _find_sign(completion: str):
    """Locate the final written sign in the completion.
    Returns (char_offset_in_completion, written_sign, strategy) or (None, None, reason).
    """
    # S1: sign inside the last \boxed{...}
    hits = list(re.finditer(r"\\boxed\{\s*([+-])", completion))
    if hits:
        m = hits[-1]
        return m.start(1), m.group(1), "S1_boxed"
    # S2: last explicitly signed number in the completion
    hits = list(re.finditer(r"([+-])\s*\d", completion))
    if hits:
        m = hits[-1]
        return m.start(1), m.group(1), "S2_last_signed_number"
    return None, None, "no_sign"


def run(model, domain, args, adapter):
    import json
    import pandas as pd
    import torch
    from config import DATA_DIR

    if domain != "arith_simple":
        print(f"[s00] note: s00 always builds arith_simple (domain arg '{domain}' ignored)")

    out_path = os.path.join(DATA_DIR, f"{model}_arith_simple_experiment_ready.xlsx")
    if os.path.exists(out_path):
        print(f"[s00] {out_path} already exists — refusing to overwrite. "
              f"Delete it manually to rebuild.")
        return out_path

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "data_shared", "arith_control_problems.json")) as f:
        problems = json.load(f)

    tok, mdl = adapter.tokenizer, adapter.model
    is_base = model.endswith("_base")
    rows, skipped = [], []

    for i, p in enumerate(problems, 1):
        if is_base:
            prompt = f"Q: {p['problem']}\nA: "
        else:
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": p["problem"]}],
                add_generation_prompt=True, tokenize=False)
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(mdl.device)
        with torch.no_grad():
            out = mdl.generate(**enc, max_new_tokens=MAX_NEW_TOKENS,
                               do_sample=False, pad_token_id=tok.eos_token_id)
        completion = tok.decode(out[0][enc["input_ids"].shape[1]:],
                                skip_special_tokens=True)
        off, written, strategy = _find_sign(completion)
        if off is None:
            skipped.append((p["id"], strategy))
            print(f"[s00] {i}/{len(problems)} {p['id']}: SKIP ({strategy})")
            continue
        prefix_len = len(prompt)
        rows.append({
            "id": p["id"],
            "wrong_sign": written,
            "correct_sign": "-" if written == "+" else "+",
            "sign_char_offset": prefix_len + off,
            "prefix_len": prefix_len,
            "full_input": prompt + completion,
            "model": completion,
            "problem_latex": p["problem"],
            "answer": p["answer"],
            "search_strategy": strategy,
            "condition": "success",
            "true_sign": p["true_sign"],
            "written_correct": written == p["true_sign"],
        })
        print(f"[s00] {i}/{len(problems)} {p['id']}: wrote '{written}' "
              f"({'ok' if written == p['true_sign'] else 'WRONG'}, {strategy})")

    df = pd.DataFrame(rows)
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_excel(out_path, index=False)
    n_wrong = int((~df["written_correct"]).sum()) if len(df) else 0
    print(f"[s00] wrote {out_path}: {len(df)} cases "
          f"({n_wrong} sign errors, {len(skipped)} skipped: {skipped})")
    return out_path
