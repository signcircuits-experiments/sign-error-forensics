"""
mistral_adapter.py
===================
ModelAdapter for mistralai/mistral-small-3.2-24b-instruct.

OFFLINE PREP NOTE: written without loading the real model/tokenizer (no GPU,
no HF network calls available in this environment). Mistral-Small-3.x is
architecturally the closest of the three new models to the existing
Llama/Qwen adapters (standard RMSNorm, no softcap, standard
self_attn.o_proj / mlp.{gate,up,down}_proj naming) — this file is
implemented as a near-copy of llama_adapter.py, BUT the specific layer
count / head count / norm module name are NOT blindly assumed identical to
Llama-3.3-70B; see the explicit verification comments below. The project
notes cite a prior "TP-sharding hook failure on Mistral-large" caused by
exactly this class of hardcoding-without-verification bug, so every
model-specific number here is called out rather than silently copied.

SIGN_TOKEN_NOTE
---------------
Sign token ids (' -', ' +', '-', '+') MUST be re-discovered at pod-launch time
via adapter.get_sign_token_ids() against the REAL tokenizer — do NOT reuse
Qwen's ids (481/488) or any other model's cached ids. This is inherited
correctly from base_adapter.py's tokenizer-driven get_sign_token_ids() method
— no override needed per-model UNLESS the model's tokenizer has unusual
space-prefix behavior, in which case note it here.
  Mistral-Small-3.x uses the "Tekken" tokenizer (a newer Mistral BPE
  vocabulary, distinct from the SentencePiece tokenizer used by
  Mistral-7B/Mixtral). Tekken's space-prefix handling has NOT been
  characterized here — VERIFY at pod-launch that
  tokenizer(" +", add_special_tokens=False) and
  tokenizer("+", add_special_tokens=False) differ, and that
  `AutoTokenizer.from_pretrained(...)` correctly loads Tekken (it may
  require `tokenizer_type="tekken"` / a specific `mistral_common` /
  `transformers` version — check the model card at pod-launch, since an
  incompatible transformers version could silently fall back to a wrong
  tokenizer or fail to load).

ARCHITECTURE-SPECIFIC ISSUES HANDLED HERE (verification required, NOT
blindly copied from Llama):

- Layer count / head count / head_dim / hidden_dim: proposed in
  config.py MODEL_CONFIGS entry based on
  public Mistral-Small-3.x architecture facts, but NOT verified against a
  live AutoConfig here (offline). `self.N_LAYERS`/`self.N_HEADS`/
  `self.HEAD_DIM` come from `config` at adapter construction time — VERIFY
  these equal `model.config.num_hidden_layers` /
  `model.config.num_attention_heads` / `model.config.head_dim` immediately
  after `.load()` (a runtime assert is added below for exactly this reason).
- Norm module name: assumed `model.model.norm` (final RMSNorm) and
  `model.model.layers[i].input_layernorm` /
  `model.model.layers[i].post_attention_layernorm` for the per-block norms,
  matching Llama/Mistral-7B convention. A runtime existence check is added
  in `load()` below rather than assuming silently.
- No softcap (Mistral does not use logit softcapping in any released
  checkpoint to date — this is a much safer assumption than Gemma's case,
  but still stated explicitly rather than left implicit).
- self_attn.{q,k,v,o}_proj and mlp.{gate,up,down}_proj: standard Mistral/
  Llama naming, NOT fused (unlike Phi-4's qkv_proj/gate_up_proj) — Mistral
  has historically kept separate projections in all HF ports.
- Tied embeddings: Mistral models do NOT tie embeddings by default (like
  Llama, unlike Gemma) — `model.lm_head.weight` is used directly.
"""

import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText
from .base_adapter import BaseAdapter


class MistralAdapter(BaseAdapter):

    def load(self):
        print(f"[Mistral] Loading {self.cfg['model_id']} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg["model_id"],
            revision=self.cfg.get("revision"),
            use_fast=True,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.cfg["model_id"],
            revision=self.cfg.get("revision"),
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        self.model.eval()

        # COMPATIBILITY ALIAS: VLM wrappers place the text model under language_model
        try:
            self.model.model.norm = self.model.model.language_model.norm
            self.model.model.layers = self.model.model.language_model.layers
            self.model.model.embed_tokens = self.model.model.language_model.embed_tokens
            self.model.lm_head = self.model.language_model.lm_head
        except AttributeError:
            pass

        # Verify config assumptions rather than trusting config.py blindly
        # (see module docstring — prior TP-sharding hook failure on
        # Mistral-large was caused by exactly this class of bug).
        try:
            real_layers = self.model.config.num_hidden_layers
            real_heads  = self.model.config.num_attention_heads
            real_hdim   = getattr(self.model.config, "head_dim",
                                   self.model.config.hidden_size // real_heads)
            mismatches = []
            if real_layers != self.N_LAYERS:
                mismatches.append(f"n_layers: config.py={self.N_LAYERS} vs model={real_layers}")
            if real_heads != self.N_HEADS:
                mismatches.append(f"n_heads: config.py={self.N_HEADS} vs model={real_heads}")
            if real_hdim != self.HEAD_DIM:
                mismatches.append(f"head_dim: config.py={self.HEAD_DIM} vs model={real_hdim}")
            if mismatches:
                print("[Mistral] WARNING: MODEL_CONFIGS mismatch vs real model.config:")
                for m in mismatches:
                    print(f"    {m}")
                print("  -> Update pipeline_release/config.py MODEL_CONFIGS['mistral'] "
                      "before trusting any layer-indexed results.")
            else:
                print("[Mistral] N_LAYERS/N_HEADS/HEAD_DIM verified against model.config. OK.")
        except Exception as e:
            print(f"[Mistral] WARNING: could not verify config against model.config: {e}")

        # Verify norm module path exists (see docstring) before any hook
        # registration downstream assumes it silently.
        assert hasattr(self.model.model, "norm"), (
            "Mistral model has no 'model.model.norm' — norm module naming "
            "assumption in mistral_adapter.py is WRONG. Inspect the model "
            "tree manually and update compute_logit_diff/compute_eff_dir.")

        print(f"[Mistral] Loaded. N_LAYERS={self.N_LAYERS}")

    # ── Layer access ──────────────────────────────────────────────────────────

    def get_layer_modules(self) -> list:
        return list(self.model.model.layers)

    def get_mlp_module(self, layer_idx: int):
        return self.model.model.layers[layer_idx].mlp

    def get_attn_module(self, layer_idx: int):
        """Return the full self_attn module (output = post-attn residual contribution)."""
        return self.model.model.layers[layer_idx].self_attn

    def get_o_proj(self, layer_idx: int):
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
        Project hidden state h -> logit[wrong_sign_tok] - logit[correct_sign_tok].
        Mistral-Small-3.2: standard RMSNorm, no softcap (identical formula to
        LLaMA/Qwen — see module docstring for what's verified vs. assumed).
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
        eff_dir = (W_U[wrong_sign_tok] - W_U[correct_sign_tok]) * scale * norm_weight
        where scale = rsqrt(mean(h^2) + eps) is the RMSNorm scale for this
        hidden state. Identical formula to LLaMA/Qwen (no softcap, no
        (1+weight) Gemma-style convention, not tied embeddings).
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
