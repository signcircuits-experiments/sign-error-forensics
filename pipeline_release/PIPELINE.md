# Sign-Circuit Pipeline

One driver, stages s00–s09, any model. Verified experiment logic is unchanged —
only the model-specific numbers (layers, paths) are parameterized.

## Quick start

```bash
python run.py --list                                                  # show stages
python run.py --model llama --domain det_4x4_error --stages 01,02 --limit 5   # smoke
python run.py --model llama --domain det_4x4_error --stages 01,02             # full
python run.py --model llama --domain det_4x4_error --stages 03                # local, no GPU
```

## Input contract

One experiment_ready file per (model, domain):

```
{SIGN_DATA_DIR}/{model}_{domain}_experiment_ready.xlsx
```

Required columns: `id`, `wrong_sign`, `correct_sign`, `sign_char_offset`,
`prefix_len`, `full_input`. (Legacy Qwen filenames are mapped automatically in
`config.py`.) Data preparation lives OUTSIDE the pipeline — each model brings its
own experiment_ready file.

## Output layout

```
{SIGN_RESULTS_DIR}/{Model}/{domain}/{v1_zero_ablation|v2_mean_ablation}/...
{SIGN_RESULTS_DIR}/{Model}/{domain}/targets.json
```

Env vars: `SIGN_DATA_DIR` (default `/workspace/data`), `SIGN_RESULTS_DIR`
(default `/workspace/results`), `SIGN_SKIP_VERSION_CHECK=1` to silence the
library version warning.

## Stages

| num | name            | GPU | what it does                                                        | output |
|-----|-----------------|-----|---------------------------------------------------------------------|--------|
| s00 | arith_control   | GPU | Builds the arithmetic control set: the model solves the SAME 48 problems (data_shared/arith_control_problems.json) and its own completions become the arith_simple experiment_ready file | {model}_arith_simple_experiment_ready.xlsx |
| s01 | logit_lens      | GPU | ExpA: sign logit-diff at every layer; finds peak/flip layers        | expA_logit_lens.json |
| s02 | dla             | GPU | ExpB: per-layer MLP/attn direct logit attribution (`--matched`, `--head-layers`) | expB_dla.json |
| s03 | derive_targets  | —   | Derives habit/check/control layers + peak groups from s01+s02       | targets.json |
| s04 | projections     | GPU | Residual/weight projections onto the sign direction at target layers | expProj JSONs |
| s05 | ablations       | GPU | ExpD′/E′ (and C′ via `--with-heads`): knock out habit/check/control MLPs | ablation JSONs |
| s06 | bias_subtraction| GPU | ExpG′: subtract MLP bias contribution                                | expG JSONs |
| s07 | patching        | GPU | ExpF′ + reverse + sign-matched: patch residuals from correct donors  | patching JSONs |
| s08 | steering        | GPU | ExpC2/ctrl_mag/generate_flipped (error) or c2_on_correct (correct)   | steering JSONs |
| s09 | judge           | GPU | Arm A/B + crossdomain judging of steered generations                 | judge JSONs |

## targets.json workflow (critical)

Layer numbers are NEVER hardcoded. For a new model:

1. Run s01 + s02 (full set).
2. Run s03 → writes `targets.json` (habit_layers, check_layer, control_layers,
   late window, emb/late peak split). `human_reviewed` is `false`.
3. **A human reads targets.json, edits if needed, sets `human_reviewed: true`.**
4. Only then run s04–s09. Stages warn if `human_reviewed` is false.

s03 refuses to overwrite an existing targets.json without `--force`.

## Subsetting and resume

Any GPU stage: `--ids q1,q2` | `--range 1-25` (1-based inclusive) | `--limit 5`.
Already-computed question ids are skipped automatically; partial runs land in
`*.part.json` and are merged in. `--no-resume` recomputes everything. The model
loads ONCE per invocation and is shared across selected stages — so 200
questions are processed one at a time with restart safety, never lost mid-run.

## Deriving release JSONs

The released `{domain}_habit_ablation.json` / `{domain}_joint_ablation.json`
files are per-case reshapes of the raw s05 outputs (expD'/expE'), produced by
`tools/derive_s05_release_json.py` (joins written_sign/sign_subtype from the
s01 expA output; `--verify` re-derives and compares against a released file).
The raw expD'/expE' files remain ground truth.

## Adding a model

1. Add an entry to `MODEL_CONFIGS` in `config.py` (model_id, results_name).
2. If loading/hooking differs, add an adapter in `adapters/` and register it in
   `adapters/__init__.py` (Qwen and Llama adapters already exist). Adapters for
   gemma-3-27b / phi-4 / mistral-small are included for forthcoming releases;
   only the Qwen adapter is exercised by the current data drop.
3. Produce `{model}_{domain}_experiment_ready.xlsx`.
4. Run stages 01 → 09 in order. Nothing else changes.

**Mandatory control (every model, including base variants):** run
`--domain arith_simple --stages 00` then `--stages 01,02` on it. Every model
answers the identical 48 arithmetic problems, so the late-layer (L75–79) DLA
profiles are directly comparable across models and against base variants
(general sign circuit vs task-specific circuit; pretraining-origin evidence).
s00 never overwrites an existing arith file — delete it manually to rebuild.

## Recommended launch order for a new model (e.g. Llama, today)

```bash
# 1. smoke: 5 questions per domain
python run.py --model llama --domain det_4x4_error --stages 01,02 --limit 5
# 2. full runs per domain
python run.py --model llama --domain det_4x4_error --stages 01,02
# 3. derive + HUMAN-REVIEW targets.json, set human_reviewed=true
python run.py --model llama --domain det_4x4_error --stages 03
# 4. then projections
python run.py --model llama --domain det_4x4_error --stages 04
# HOLD s05–s09 until targets.json is reviewed. Never run them blind.
```
