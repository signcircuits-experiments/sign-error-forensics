"""
llama_adapter.py
================
ModelAdapter for LLaMA-3.3-70B-Instruct.
Standard RMSNorm, no softcap, full attention architecture.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from .base_adapter import BaseAdapter


class LLaMAAdapter(BaseAdapter):

    def load(self):
        print(f"[LLaMA] Loading {self.cfg['model_id']} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg["model_id"],
            use_fast=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg["model_id"],
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        self.model.eval()
        print(f"[LLaMA] Loaded. N_LAYERS={self.N_LAYERS}")

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
        Project hidden state h → logit[wrong_sign_tok] - logit[correct_sign_tok].
        LLaMA: standard RMSNorm, no softcap.
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
        Compute DLA effective direction.
        LLaMA: eff_dir = (W_U[wrong_sign_tok] - W_U[correct_sign_tok]) * scale * norm_weight
        where scale = rsqrt(mean(h²) + eps) is the RMSNorm scale for this hidden state.
        """
        norm_dev = next(self.model.model.norm.parameters()).device
        h = h_final.to(norm_dev).to(torch.bfloat16)

        # RMSNorm scale at this hidden state
        eps  = self.model.config.rms_norm_eps
        scale = torch.rsqrt(h.float().pow(2).mean() + eps)  # scalar

        # LN weight
        norm_w = self.model.model.norm.weight.float()

        # Unembedding rows
        W_U = self.model.lm_head.weight.float()  # [vocab, hidden]
        diff = (W_U[wrong_sign_tok] - W_U[correct_sign_tok])   # [hidden]

        # eff_dir: what component output must dot with to get logit contribution
        eff_dir = diff * scale * norm_w  # [hidden]
        return eff_dir.cpu()
