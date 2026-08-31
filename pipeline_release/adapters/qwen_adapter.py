"""
qwen_adapter.py
===============
ModelAdapter for Qwen2.5-72B-Instruct.
Same architecture as LLaMA (standard RMSNorm, no softcap, GQA).
Only differences: model_id and chat template (handled in base via response_marker).
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .base_adapter import BaseAdapter


class QwenAdapter(BaseAdapter):

    def load(self):
        print(f"[Qwen] Loading {self.cfg['model_id']} ...")
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
        print(f"[Qwen] Loaded. N_LAYERS={self.N_LAYERS}")

    # ── Layer access ──────────────────────────────────────────────────────────

    def get_layer_modules(self) -> list:
        return list(self.model.model.layers)

    def get_mlp_module(self, layer_idx: int):
        return self.model.model.layers[layer_idx].mlp

    def get_attn_module(self, layer_idx: int):
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
        Qwen: logit[wrong_sign_tok] - logit[correct_sign_tok]. Standard RMSNorm, no softcap — identical to LLaMA.
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
        Qwen: eff_dir = (W_U[wrong_sign_tok] - W_U[correct_sign_tok]) * scale * norm_weight.
        No Jacobian correction needed. GQA does not affect eff_dir (lm_head operates on full hidden dim).
        """
        norm_dev = next(self.model.model.norm.parameters()).device
        h = h_final.to(norm_dev).to(torch.bfloat16)

        eps   = self.model.config.rms_norm_eps
        scale = torch.rsqrt(h.float().pow(2).mean() + eps)
        norm_w = self.model.model.norm.weight.float()
        W_U   = self.model.lm_head.weight.float()
        diff  = (W_U[wrong_sign_tok] - W_U[correct_sign_tok])
        eff_dir = diff * scale * norm_w
        return eff_dir.cpu()
