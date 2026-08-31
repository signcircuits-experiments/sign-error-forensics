"""
gemma_adapter.py
================
ModelAdapter for google/gemma-3-27b-it.

OFFLINE PREP NOTE: written without loading the real model/tokenizer (no GPU,
no HF network calls available in this environment). All module-path and
architecture claims below are based on public Gemma-3 / HF `Gemma3ForCausalLM`
implementation knowledge as of training time and MUST be spot-checked against
`AutoConfig.from_pretrained(...)` / a real `print(model)` at pod-launch time
before trusting any causal-intervention numbers.

SIGN_TOKEN_NOTE
---------------
Sign token ids (' -', ' +', '-', '+') MUST be re-discovered at pod-launch time
via adapter.get_sign_token_ids() against the REAL tokenizer — do NOT reuse
Qwen's ids (481/488) or any other model's cached ids. This is inherited
correctly from base_adapter.py's tokenizer-driven get_sign_token_ids() method
— no override needed per-model UNLESS the model's tokenizer has unusual
space-prefix behavior, in which case note it here.
  Gemma's SentencePiece tokenizer uses "▁" (U+2581) as an explicit
  space-marker prefix baked into token strings (not a literal leading space
  character), so the has_space heuristic in base_adapter.get_sign_token_ids
  (checking `full_input[offset-1] == " "`) should still work at the STRING
  level (tokenizer() is called on plain python strings " +"/"+", and Gemma's
  fast tokenizer will internally map the leading space to "▁" correctly) —
  but this has NOT been empirically verified for Gemma-3's fast tokenizer.
  VERIFY at pod-launch: print(tokenizer(" +", add_special_tokens=False)) and
  tokenizer("+", add_special_tokens=False) and confirm they differ.

ARCHITECTURE-SPECIFIC ISSUES HANDLED HERE (all low-confidence, verify at
pod-launch — see comments inline at each point):

(a) Logit softcapping — Gemma-2 used `final_logit_softcapping` (tanh-based
    softcap on the final logits) and `attn_logit_softcapping`. Gemma-3
    REMOVED both softcaps (replaced by QK-norm for training stability
    instead) per the Gemma-3 technical report and the HF `Gemma3Config`
    class, which has no `final_logit_softcapping` field (unlike
    `Gemma2Config`). Basis for this decision: Gemma-3's public architecture
    docs / HF `modeling_gemma3.py` source, which implements
    `Gemma3ForCausalLM.forward` as a *plain* linear lm_head projection with
    no softcap wrapper (compare to `Gemma2ForCausalLM.forward`, which calls
    `torch.tanh` after dividing by `final_logit_softcapping`). We therefore
    set `HAS_SOFTCAP = False` below and implement compute_logit_diff /
    compute_eff_dir WITHOUT a softcap Jacobian correction. A human should
    flip `HAS_SOFTCAP = True` and un-comment the Jacobian-correction branch
    below if `model.config.final_logit_softcapping is not None` at
    pod-launch time.

(b) Tied embeddings — Gemma models tie the unembedding matrix W_U to the
    input embedding matrix (`tie_word_embeddings=True`). HF's
    `AutoModelForCausalLM` machinery normally keeps `model.lm_head.weight`
    pointing at the SAME tensor as `model.model.embed_tokens.weight` after
    tying, so `self.model.lm_head.weight` should still work — but to be
    safe (and match the spec's explicit request) we access the embedding
    matrix directly via `self.model.model.language_model.embed_tokens.weight`, with a
    fallback to `get_output_embeddings()` for robustness. VERIFY at
    pod-launch: `(model.lm_head.weight == model.model.embed_tokens.weight).all()`.

(c) Norm layout — Gemma-2/3 use a "sandwich" norm layout per block (NOT the
    Llama pre-norm-only layout):
        residual = h
        h = input_layernorm(h)
        h = self_attn(h)
        h = post_attention_layernorm(h)     # <- EXTRA norm, no Llama equivalent
        h = residual + h
        residual = h
        h = pre_feedforward_layernorm(h)    # <- EXTRA norm, no Llama equivalent
        h = mlp(h)
        h = post_feedforward_layernorm(h)   # <- EXTRA norm, no Llama equivalent
        h = residual + h
    Expected HF module tree (Gemma3ForCausalLM, text-only checkpoint):
        model.model.embed_tokens                          Embedding
        model.model.layers[i].self_attn.{q,k,v,o}_proj     Linear
        model.model.layers[i].self_attn.q_norm/k_norm      Gemma3RMSNorm (QK-norm, new in v3)
        model.model.layers[i].mlp.{gate,up,down}_proj      Linear
        model.model.layers[i].input_layernorm              Gemma3RMSNorm
        model.model.layers[i].post_attention_layernorm     Gemma3RMSNorm
        model.model.layers[i].pre_feedforward_layernorm    Gemma3RMSNorm
        model.model.layers[i].post_feedforward_layernorm   Gemma3RMSNorm
        model.model.norm                                   Gemma3RMSNorm (final)
        model.lm_head                                      Linear (tied)
    compute_eff_dir/compute_logit_diff below use `model.model.norm` for the
    FINAL projection (correct — this is analogous to Llama's `model.norm`).
    *** CRITICAL CAVEAT for DLA-style experiments (expB/expD/expE/expC-style
    head ablation) that hook `get_mlp_module`/`get_attn_module` raw outputs:
    for Gemma, the raw self_attn/mlp module output is NOT what gets added to
    the residual stream — it must first pass through
    post_attention_layernorm / post_feedforward_layernorm. Any DLA computed
    as dot(raw_module_output, eff_dir) will therefore be WRONG for Gemma
    unless the calling experiment code is extended to also apply the
    corresponding post-norm before the dot product. This adapter does NOT
    silently patch that in (get_attn_module/get_mlp_module return the raw
    modules, matching the existing BaseAdapter contract used by
    expB_dla.py/expD_prime.py/mean_ablation_utils.py) — flagged here as a
    LOWEST CONFIDENCE / must-verify item. This adapter is included for a
    forthcoming release and is untested against the current data drop.
    Gemma's RMSNorm ALSO differs numerically from Llama's: HF's
    `Gemma3RMSNorm.forward` computes `output * (1.0 + weight.float())`
    (note the `1 +`), not `output * weight` like LlamaRMSNorm. This adapter
    implements that `(1 + weight)` convention explicitly below.
    Also: Gemma-3 additionally scales the embedding output by
    `sqrt(hidden_size)` right after `embed_tokens` (a "normalizer" multiply
    inside `Gemma3TextModel.forward`) before it enters layer 0. This is
    included in get_embedding_output below since some baseline/no-ablation
    analyses treat the embedding as a residual-stream reference point.
"""

import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .base_adapter import BaseAdapter

# See docstring (a) above — flip to True + implement the softcap Jacobian
# branch below if pod-launch AutoConfig shows final_logit_softcapping is set.
HAS_SOFTCAP = False


class GemmaAdapter(BaseAdapter):

    def load(self):
        print(f"[Gemma] Loading {self.cfg['model_id']} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg["model_id"],
            revision=self.cfg.get("revision"),
            use_fast=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg["model_id"],
            revision=self.cfg.get("revision"),
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        self.model.eval()
        # Sanity check tied embeddings — verify at pod-launch (see docstring b).
        try:
            tied = torch.equal(self.model.lm_head.weight.data,
                                self.model.model.language_model.embed_tokens.weight.data)
            print(f"[Gemma] lm_head tied to embed_tokens: {tied}")
        except Exception as e:
            print(f"[Gemma] WARNING: could not verify tied-embedding assumption: {e}")
            
        # COMPATIBILITY ALIAS: experiment scripts hardcode `adapter.model.model.norm`.
        # For Gemma 3, this is located at `model.model.language_model.norm`.
        try:
            self.model.model.norm = self.model.model.language_model.norm
            self.model.model.layers = self.model.model.language_model.layers
        except AttributeError:
            pass # fallback if some versions don't need it
            
        print(f"[Gemma] Loaded. N_LAYERS={self.N_LAYERS}  HAS_SOFTCAP={HAS_SOFTCAP}")

    # ── Layer access ──────────────────────────────────────────────────────────

    def get_layer_modules(self) -> list:
        return list(self.model.model.language_model.layers)

    def get_mlp_module(self, layer_idx: int):
        """
        Returns post_feedforward_layernorm so DLA captures the exact quantity
        added to the residual stream (avoiding 100x scale errors from raw mlp).
        """
        return self.model.model.language_model.layers[layer_idx].post_feedforward_layernorm

    def get_attn_module(self, layer_idx: int):
        """
        Returns post_attention_layernorm so DLA captures the exact quantity
        added to the residual stream.
        """
        return self.model.model.language_model.layers[layer_idx].post_attention_layernorm

    def get_o_proj(self, layer_idx: int):
        return self.model.model.language_model.layers[layer_idx].self_attn.o_proj

    def get_embedding_output(self, full_ids: torch.Tensor,
                              probe_tok: int) -> torch.Tensor:
        """
        Gemma scales the embedding output by sqrt(hidden_size) before it
        enters layer 0 (see docstring c). Included here so this matches what
        actually flows into the residual stream.
        """
        with torch.no_grad():
            embed = self.model.model.language_model.embed_tokens(full_ids)
            normalizer = torch.tensor(self.cfg["hidden_dim"] ** 0.5,
                                       dtype=embed.dtype, device=embed.device)
            embed = embed * normalizer
        return embed[0, probe_tok, :].detach().float().cpu()

    # ── Core computation ──────────────────────────────────────────────────────

    def _gemma_rmsnorm(self, h: torch.Tensor, norm_module, eps: float) -> torch.Tensor:
        """
        Gemma's RMSNorm applies (1 + weight), NOT weight (see docstring c).
        h: [hidden] float tensor (already on norm device).
        """
        h_f = h.float()
        scale = torch.rsqrt(h_f.pow(2).mean(-1, keepdim=True) + eps)
        normed = h_f * scale
        weight = norm_module.weight.float()
        return normed * (1.0 + weight)

    def compute_logit_diff(self, h: torch.Tensor,
                           wrong_sign_tok: int, correct_sign_tok: int) -> float:
        """
        Gemma-3: RMSNorm with (1+weight) scaling, then tied lm_head.
        No final-logit softcap applied (HAS_SOFTCAP=False, see docstring a).
        If a human sets HAS_SOFTCAP=True after verifying model.config, this
        function must be extended to apply:
            logits = softcap * tanh(raw_logits / softcap)
        before taking the (wrong - correct) difference.
        """
        norm_module = self.model.model.language_model.norm
        norm_dev = next(norm_module.parameters()).device
        eps = self.model.config.text_config.rms_norm_eps

        h_dev = h.to(norm_dev).float()
        h_norm = self._gemma_rmsnorm(h_dev, norm_module, eps).to(torch.bfloat16)

        with torch.no_grad():
            logits = self.model.lm_head(h_norm.unsqueeze(0).unsqueeze(0))[0, 0, :].float()

        if HAS_SOFTCAP:
            softcap = getattr(self.model.config, "final_logit_softcapping", None)
            if softcap:
                logits = softcap * torch.tanh(logits / softcap)

        return (logits[wrong_sign_tok] - logits[correct_sign_tok]).item()

    def compute_eff_dir(self, wrong_sign_tok: int, correct_sign_tok: int,
                         h_final: torch.Tensor) -> torch.Tensor:
        """
        eff_dir = (W_U[wrong] - W_U[correct]) * RMSNorm_scale * (1 + norm_weight)

        If HAS_SOFTCAP is later set True, this must be extended with a
        softcap Jacobian correction: since logits = c*tanh(raw/c), the local
        linearization at the operating point raw0 is
            d(logits)/d(raw) = 1 - tanh(raw0/c)^2
        so eff_dir should be scaled per-target-token by that Jacobian factor
        evaluated at the actual (uncapped) logit for wrong/correct tokens —
        NOT a single global scalar, since it differs per token. This is
        deliberately NOT implemented (HAS_SOFTCAP=False) — see docstring (a).
        """
        norm_module = self.model.model.language_model.norm
        norm_dev = next(norm_module.parameters()).device
        eps = self.model.config.text_config.rms_norm_eps

        h = h_final.to(norm_dev).float()
        scale = torch.rsqrt(h.pow(2).mean() + eps)  # scalar

        # (1 + weight), per docstring (c)
        norm_w = 1.0 + norm_module.weight.float()

        # Tied embedding matrix used as W_U — see docstring (b).
        try:
            W_U = self.model.lm_head.weight.float()
        except AttributeError:
            W_U = self.model.get_output_embeddings().weight.float()

        diff = (W_U[wrong_sign_tok] - W_U[correct_sign_tok])  # [hidden]

        eff_dir = diff * scale * norm_w  # [hidden]

        if HAS_SOFTCAP:
            # Placeholder — NOT implemented. See method docstring above.
            print("[Gemma] WARNING: HAS_SOFTCAP=True but no Jacobian correction "
                  "implemented in compute_eff_dir — results will be WRONG. "
                  "Implement the per-token tanh Jacobian before trusting DLA.")

        return eff_dir.cpu()
