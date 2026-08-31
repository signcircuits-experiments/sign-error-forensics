"""
phi_adapter.py
==============
ModelAdapter for microsoft/phi-4.

OFFLINE PREP NOTE: written without loading the real model/tokenizer (no GPU,
no HF network calls available in this environment). Module-path claims below
are based on known Phi-3/Phi-4 HF implementation conventions
(`Phi3ForCausalLM` — Phi-4's config.json uses `"model_type": "phi3"`, i.e.
Phi-4 reuses the Phi-3 modeling code) and MUST be verified against the
actual `AutoConfig`/`print(model)` at pod-launch time before trusting any
causal-intervention numbers. The `--dry-run` mode in exp_head_ablation.py
(Part 3) is specifically designed to catch a wrong module path here (e.g.
the "960/960 no-capture" class of failure mentioned in the project notes)
before any GPU time is spent on a real forward pass.

SIGN_TOKEN_NOTE
---------------
Sign token ids (' -', ' +', '-', '+') MUST be re-discovered at pod-launch time
via adapter.get_sign_token_ids() against the REAL tokenizer — do NOT reuse
Qwen's ids (481/488) or any other model's cached ids. This is inherited
correctly from base_adapter.py's tokenizer-driven get_sign_token_ids() method
— no override needed per-model UNLESS the model's tokenizer has unusual
space-prefix behavior, in which case note it here.
  Phi-4 uses a GPT-4o-style tiktoken-derived BPE vocabulary (o200k-based),
  distinct from Llama/Qwen/Gemma's SentencePiece-family tokenizers. BPE
  vocabularies of this kind commonly have a genuinely different token for a
  leading-space "+"/"-" vs. no-space "+"/"-" (e.g. GPT-style " +" as its own
  token), so the has_space heuristic in base_adapter.get_sign_token_ids
  should work, but the actual TOKEN IDS will differ substantially from
  Qwen/Llama's ids and possibly even the multi-token-vs-single-token
  behavior could differ. VERIFY at pod-launch: print both
  tokenizer(" +", add_special_tokens=False) and
  tokenizer("+", add_special_tokens=False) and confirm plus_tok != minus_tok.

ARCHITECTURE-SPECIFIC ISSUES HANDLED HERE:

Module naming under model.layers[i] — Phi-3/Phi-4 use FUSED projections that
differ from Llama's separate q_proj/k_proj/v_proj and gate_proj/up_proj:
    model.model.layers[i].self_attn.qkv_proj      Linear (fused Q+K+V)
    model.model.layers[i].self_attn.o_proj        Linear (NOT fused — same
                                                   name/role as Llama's o_proj)
    model.model.layers[i].mlp.gate_up_proj        Linear (fused gate+up)
    model.model.layers[i].mlp.down_proj           Linear (NOT fused)
    model.model.layers[i].input_layernorm         Phi3RMSNorm
    model.model.layers[i].post_attention_layernorm Phi3RMSNorm
    model.model.norm                              Phi3RMSNorm (final)
    model.model.embed_tokens                      Embedding
    model.lm_head                                 Linear (NOT tied, standard
                                                   Phi-3/Phi-4 default)
These exact names (`qkv_proj`, `gate_up_proj`, `o_proj`, `down_proj`) are
based on known Phi-3-mini/medium HF `modeling_phi3.py` conventions extended
to Phi-4 (same architecture family). THIS MUST BE VERIFIED against the real
Phi-4 checkpoint's module tree at pod-launch — Microsoft has in the past
made small naming changes between Phi generations, and Phi-4 could differ.
Because `o_proj` is (per this convention) NOT fused, `get_o_proj` below
mirrors Llama's `self_attn.o_proj` path — this is the single most important
assumption to verify (all of Part 3's head-ablation hooking depends on it).

Norm layout — standard Llama-style pre-norm only (no Gemma-style sandwich
norm, no softcap). This is the "boring" case: RMSNorm(h) -> lm_head, same
formula as LlamaAdapter/QwenAdapter.

Tied embeddings — Phi-3/Phi-4 are NOT believed to tie embeddings by default
(unlike Phi-3-mini's optional weight tying and unlike Gemma). This adapter
uses `model.lm_head.weight` directly (separate matrix), matching
Llama/Qwen. VERIFY at pod-launch:
`model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()`
— if True, tying IS in effect and this comment should be updated.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .base_adapter import BaseAdapter


class PhiAdapter(BaseAdapter):

    def load(self):
        print(f"[Phi] Loading {self.cfg['model_id']} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg["model_id"],
            use_fast=True,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg["model_id"],
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        self.model.eval()
        # Sanity check the fused-projection assumption documented above —
        # fail loudly rather than silently hooking the wrong module.
        try:
            layer0 = self.model.model.layers[0]
            assert hasattr(layer0.self_attn, "o_proj"), (
                "Phi-4 self_attn has no 'o_proj' — module naming assumption "
                "in phi_adapter.py is WRONG. Inspect layer0.self_attn "
                "manually and update get_o_proj()/get_attn_module().")
            print(f"[Phi] Verified self_attn.o_proj exists on layer 0. "
                  f"qkv_proj present: {hasattr(layer0.self_attn, 'qkv_proj')}")
        except Exception as e:
            print(f"[Phi] WARNING: module-path sanity check failed: {e}")
        print(f"[Phi] Loaded. N_LAYERS={self.N_LAYERS}")

    # ── Layer access ──────────────────────────────────────────────────────────

    def get_layer_modules(self) -> list:
        return list(self.model.model.layers)

    def get_mlp_module(self, layer_idx: int):
        """Returns the full mlp module (gate_up_proj + down_proj fused inside)."""
        return self.model.model.layers[layer_idx].mlp

    def get_attn_module(self, layer_idx: int):
        """Returns the full self_attn module (qkv_proj fused, o_proj separate)."""
        return self.model.model.layers[layer_idx].self_attn

    def get_o_proj(self, layer_idx: int):
        """
        ASSUMED path: self_attn.o_proj (NOT fused into qkv_proj — see module
        docstring). This is the #1 thing to verify at pod-launch time; a
        wrong path here will silently produce a no-op hook (0 captures)
        rather than a crash, which is exactly the failure class the
        --dry-run mode in exp_head_ablation.py is built to catch.
        """
        return self.model.model.layers[layer_idx].self_attn.o_proj

    def get_embedding_output(self, full_ids: torch.Tensor,
                              probe_tok: int) -> torch.Tensor:
        with torch.no_grad():
            embed = self.model.model.embed_tokens(full_ids)
        return embed[0, probe_tok, :].detach().float().cpu()

    # ── Core computation ──────────────────────────────────────────────────────

    def compute_logit_diff(self, h: torch.Tensor,
                           wrong_sign_tok: int, correct_sign_tok: int) -> float:
        """
        Phi-4: standard RMSNorm (no (1+weight) convention like Gemma, no
        softcap) -> lm_head. Same formula as LLaMA/Qwen.
        """
        norm_dev = next(self.model.model.norm.parameters()).device
        h_dev = h.to(norm_dev).to(torch.bfloat16).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            h_norm = self.model.model.norm(h_dev)
            logits  = self.model.lm_head(h_norm)[0, 0, :].float()
        return (logits[wrong_sign_tok] - logits[correct_sign_tok]).item()

    def compute_eff_dir(self, wrong_sign_tok: int, correct_sign_tok: int,
                         h_final: torch.Tensor) -> torch.Tensor:
        """
        eff_dir = (W_U[wrong] - W_U[correct]) * RMSNorm_scale * norm_weight
        Standard formula (no Gemma-style (1+weight), no softcap Jacobian).
        Uses model.lm_head.weight directly (assumed NOT tied — verify at
        pod-launch, see module docstring).
        """
        norm_dev = next(self.model.model.norm.parameters()).device
        h = h_final.to(norm_dev).to(torch.bfloat16)

        eps  = self.model.config.rms_norm_eps
        scale = torch.rsqrt(h.float().pow(2).mean() + eps)

        norm_w = self.model.model.norm.weight.float()
        W_U   = self.model.lm_head.weight.float()
        diff  = (W_U[wrong_sign_tok] - W_U[correct_sign_tok])
        eff_dir = diff * scale * norm_w
        return eff_dir.cpu()
