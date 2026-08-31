"""
pipeline/config.py
==================
Single configuration for the sign-circuit pipeline.

Design rules:
  * NO hardcoded target layers here. Per-model/per-domain target layers live in
    targets.json, written by stage s03 (derive targets) and reviewed by a human.
  * Data files follow ONE naming convention:
        {DATA_DIR}/{model}_{domain}_experiment_ready.xlsx
    Legacy Qwen filenames are kept as explicit overrides.
  * Paths are overridable by environment variables so the same code runs on any
    pod:  SIGN_DATA_DIR, SIGN_RESULTS_DIR.
"""

import os

# ── Version check (relax with SIGN_SKIP_VERSION_CHECK=1) ─────────────────────
if not os.environ.get("SIGN_SKIP_VERSION_CHECK"):
    try:
        import transformers, torch
        print(f"[config] transformers={transformers.__version__}  torch={torch.__version__}")
        if not transformers.__version__.startswith("4.46"):
            print(f"[config] WARNING: transformers {transformers.__version__} != 4.46.x "
                  f"(reference runs used 4.46; set SIGN_SKIP_VERSION_CHECK=1 to silence)")
    except ImportError:
        pass  # local (non-GPU) stages don't need torch/transformers

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = os.environ.get("SIGN_DATA_DIR",    "/workspace/data")
RESULTS_DIR = os.environ.get("SIGN_RESULTS_DIR", "/workspace/results")

# ── Model registry ────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "qwen": {
        "model_id"        : "Qwen/Qwen2.5-72B-Instruct",
        "n_layers"        : 80,
        "n_heads"         : 64,
        "n_kv_heads"      : 8,
        "head_dim"        : 128,
        "hidden_dim"      : 8192,
        "has_softcap"     : False,
        "norm_type"       : "standard",
        "response_marker" : "<|im_start|>assistant\n",
        "model_col"       : "Qwen2.5-72B",
        "results_name"    : "Qwen",        # folder under RESULTS_DIR
    },
    "qwen_heldout": {
        "model_id"        : "Qwen/Qwen2.5-72B-Instruct",
        "n_layers"        : 80,
        "n_heads"         : 64,
        "n_kv_heads"      : 8,
        "head_dim"        : 128,
        "hidden_dim"      : 8192,
        "has_softcap"     : False,
        "norm_type"       : "standard",
        "response_marker" : "<|im_start|>assistant\n",
        "model_col"       : "Qwen2.5-72B",
        "results_name"    : "Qwen_heldout",
    },
    "qwen_base": {
        "model_id"        : "Qwen/Qwen2.5-72B",
        "n_layers"        : 80,
        "n_heads"         : 64,
        "n_kv_heads"      : 8,
        "head_dim"        : 128,
        "hidden_dim"      : 8192,
        "has_softcap"     : False,
        "norm_type"       : "standard",
        "response_marker" : "\nA: ",
        "model_col"       : "Qwen2.5-72B-Base",
        "results_name"    : "Qwen_base",
    },
    "llama": {
        "model_id"        : "meta-llama/Llama-3.3-70B-Instruct",
        "n_layers"        : 80,
        "n_heads"         : 64,
        "n_kv_heads"      : 8,
        "head_dim"        : 128,
        "hidden_dim"      : 8192,
        "has_softcap"     : False,
        "norm_type"       : "standard",
        "response_marker" : "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "model_col"       : "Llama-3.3-70B",
        "results_name"    : "Llama",
    },
    "llama_base": {
        "model_id"        : "meta-llama/Llama-3.1-70B",   # base (non-Instruct)
        "n_layers"        : 80,
        "n_heads"         : 64,
        "n_kv_heads"      : 8,
        "head_dim"        : 128,
        "hidden_dim"      : 8192,
        "has_softcap"     : False,
        "norm_type"       : "standard",
        "response_marker" : "\nA: ",
        "model_col"       : "Llama-70B-Base",
        "results_name"    : "Llama_base",
    },
    "gemma": {
        "model_id"        : "google/gemma-3-27b-it",
        "revision"        : "005ad3404e59d6023443cb575daa05336842228a",
        "n_layers"        : 62,      # LOW-MED CONFIDENCE — verify against AutoConfig at pod-launch
        "n_heads"         : 32,      # LOW-MED CONFIDENCE
        "n_kv_heads"      : 16,      # LOW-MED CONFIDENCE (GQA)
        "head_dim"        : 128,     # LOW-MED CONFIDENCE
        "hidden_dim"      : 5376,    # LOW-MED CONFIDENCE
        "has_softcap"     : False,   # Gemma-3 dropped softcap (QK-norm instead) — verify model.config.final_logit_softcapping is None
        "norm_type"       : "gemma_sandwich_1plusweight",  # NOT "standard" — see gemma_adapter.py docstring
        "response_marker" : "<start_of_turn>model\n",      # confirmed against actual full_input text
        "model_col"       : "Gemma-3",
        "results_name"    : "Gemma",
    },
    "phi": {
        "model_id"        : "microsoft/phi-4",
        "n_layers"        : 40,      # LOW-MED CONFIDENCE — verify against AutoConfig at pod-launch
        "n_heads"         : 40,      # LOW-MED CONFIDENCE
        "n_kv_heads"      : 10,      # LOW-MED CONFIDENCE (GQA)
        "head_dim"        : 128,     # LOW-MED CONFIDENCE
        "hidden_dim"      : 5120,    # LOW-MED CONFIDENCE
        "has_softcap"     : False,
        "norm_type"       : "standard",
        "response_marker" : "<|im_start|>assistant<|im_sep|>",  # confirmed against actual full_input text
        "model_col"       : "Phi-4",
        "results_name"    : "Phi",
    },
    "mistral": {
        "model_id"        : "mistralai/Mistral-Small-24B-Instruct-2501",
        "revision"        : "9527884be6e5616bdd54de542f9ae13384489724",
        "n_layers"        : 40,      # LOW-MED CONFIDENCE — verify against AutoConfig at pod-launch
        "n_heads"         : 32,      # LOW-MED CONFIDENCE
        "n_kv_heads"      : 8,       # LOW-MED CONFIDENCE (GQA)
        "head_dim"        : 128,     # LOW-MED CONFIDENCE
        "hidden_dim"      : 5120,    # LOW-MED CONFIDENCE
        "has_softcap"     : False,
        "norm_type"       : "standard",
        "response_marker" : "[/INST]",   # confirmed against actual full_input text
        "model_col"       : "Mistral-24B",
        "results_name"    : "Mistral",
    },
    # Add a new model: one entry here + one adapter file in adapters/.
}

# ── Domains ───────────────────────────────────────────────────────────────────
# A domain = one experiment_ready file for one model.
KNOWN_DOMAINS = [
    "det_3x3_error", "det_3x3_correct",
    "det_4x4_error", "det_4x4_correct",
    "det_5x5_error", "det_5x5_correct",
    "ibp_error",     "ibp_correct",
    "arith_simple",
    # New small-model domains (gemma/phi/mistral) — error-only, no correct-domain
    # counterpart exists for any of these (condition=='failure' 100% in all 5 source files).
    "determinant", "rank", "nullity", "eigenvalue", "integration_by_parts",
]

# New small models: error-only domains (no *_correct pairing exists in the source data).
NO_CORRECT_DOMAINS = {"determinant", "rank", "nullity", "eigenvalue", "integration_by_parts"}

# Error-domain <-> correct-domain pairing (needed by patching/judge stages).
DOMAIN_PAIRS = {
    "det_3x3_error": "det_3x3_correct",
    "det_4x4_error": "det_4x4_correct",
    "det_5x5_error": "det_5x5_correct",
    "ibp_error":     "ibp_correct",
}

# Domains whose expA/expB results are stored under v2_mean_ablation
# (correct-answer sets; error domains store expA/expB under v1_zero_ablation).
CORRECT_DATASETS = {
    "det_3x3_correct", "det_4x4_correct", "det_5x5_correct",
    "ibp_correct", "arith_simple",
}

# ── Data files: convention + legacy overrides ────────────────────────────────
_LEGACY_QWEN = {
    "det_5x5_error":   "qwen_5x5_experiment_ready.xlsx",
    "det_4x4_error":   "QWEN_4x4_experiment_ready.xlsx",
    "det_4x4_correct": "qwen_4x4_correct_experiment_ready.xlsx",
    "det_5x5_correct": "qwen_5x5_correct_experiment_ready.xlsx",
    "ibp_error":       "qwen_ibp_experiment_ready3.xlsx",
    "det_3x3_correct": "qwen_3x3_correct_experiment_ready.xlsx",
    "det_3x3_error":   "qwen_3x3_error_experiment_ready.xlsx",
    "ibp_correct":     "qwen_ibp_correct_experiment_ready.xlsx",
    "arith_simple":    "qwen_arith_simple_experiment_ready.xlsx",
}
_LEGACY_LLAMA = {
    "det_5x5_error":   "llama_det_5x5_error_n50_experiment_ready.xlsx",
    "det_4x4_error":   "llama_det_4x4_error_n16_experiment_ready.xlsx",
    "ibp_error":       "llama_ibp_error_n10_experiment_ready.xlsx",
    "det_4x4_correct": "llama_det_4x4_correct_set1_n80_experiment_ready.xlsx",
    "ibp_correct":     "llama_ibp_correct_n23_experiment_ready.xlsx",
}
_LEGACY = {
    "qwen":      _LEGACY_QWEN,
    "qwen_base": {"arith_simple": "qwen_arith_simple_base_experiment_ready.xlsx"},
    "llama":     _LEGACY_LLAMA,
}


def data_file(model: str, domain: str) -> str:
    """Path of the experiment_ready file for (model, domain).

    Note: arith_simple for any NEW model is built by stage s00 (arith_control)
    from the shared problem set data_shared/arith_control_problems.json and
    lands on the default convention below — no legacy entry needed.
    """
    fname = _LEGACY.get(model, {}).get(
        domain, f"{model}_{domain}_experiment_ready.xlsx")
    return os.path.join(DATA_DIR, fname)


# Backward-compatible DATA_FILES dict (experiments/ modules import this).
DATA_FILES = {
    m: {d: data_file(m, d) for d in KNOWN_DOMAINS} for m in MODEL_CONFIGS
}

# ── Result path helpers ───────────────────────────────────────────────────────
def results_root(model: str) -> str:
    return os.path.join(RESULTS_DIR, MODEL_CONFIGS[model]["results_name"])


def result_path(model: str, domain: str, fname: str, version: str = None) -> str:
    """Standard result location. version defaults to the domain's convention."""
    if version is None:
        version = "v2_mean_ablation" if domain in CORRECT_DATASETS else "v1_zero_ablation"
    p = os.path.join(results_root(model), domain, version, fname)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def targets_path(model: str, domain: str) -> str:
    """Per-model/per-domain targets.json written by s03, read by s04-s09."""
    p = os.path.join(results_root(model), domain, "targets.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


# ── Experiment parameters (pre-registered thresholds, model-free) ────────────
NOISE_FLOOR_LOGIT = 0.1     # expA flip_layer noise floor (logit units)

PUSHER_THRESHOLD_ABS    = 0.10
PUSHER_THRESHOLD_FRAC   = 0.05
PUSHER_CONFIRM_DELTA    = -0.5
PUSHER_CONFIRM_RATE     = 0.60
CORRECTOR_CONFIRM_DELTA = 0.5
CORRECTOR_CONFIRM_RATE  = 0.60

MLP_PUSHER_CONFIRM_DELTA   = -1.0
MLP_PUSHER_CONFIRM_RATE    = 0.75
MLP_CONTROL_NULL_THRESHOLD = 0.5

JOINT_MLP_CONFIRM_DELTA = -2.0
JOINT_MLP_CONFIRM_RATE  = 0.80

# G' / steering doses
ALPHA_DOSES = [0.5, 1.0, 2.0, 3.0]

# ── Legacy names some experiments/ modules still import ──────────────────────
# (kept so the verbatim experiment implementations run unmodified; the VALUES
#  are irrelevant when stages pass targets explicitly)
ABLATION_TARGETS   = {m: [] for m in MODEL_CONFIGS}
MLP_ABLATION_LAYERS = {m: [] for m in MODEL_CONFIGS}
JOINT_MLP_TARGETS   = {m: [] for m in MODEL_CONFIGS}
