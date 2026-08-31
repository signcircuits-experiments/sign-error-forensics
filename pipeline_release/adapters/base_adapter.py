"""
base_adapter.py
===============
Abstract base class for model adapters.
Each model implements these methods; experiment scripts call only adapter methods.
"""

from abc import ABC, abstractmethod
import torch


class BaseAdapter(ABC):

    def __init__(self, model_name: str, config: dict):
        self.model_name = model_name
        self.cfg = config
        self.model = None
        self.tokenizer = None
        self.N_LAYERS = config["n_layers"]
        self.N_HEADS  = config["n_heads"]
        self.HEAD_DIM = config["head_dim"]

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    @abstractmethod
    def load(self):
        """Load model and tokenizer onto GPU."""

    def unload(self):
        """Free GPU memory."""
        import gc
        del self.model
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()

    # ── Data helpers ──────────────────────────────────────────────────────────

    def get_response_start_char(self, full_input: str) -> int:
        """
        Return the character offset where the assistant response begins.
        Uses prefix_len from the dataset (pre-computed) — but also validates
        against the response_marker for safety.
        """
        marker = self.cfg["response_marker"]
        idx = full_input.find(marker)
        if idx == -1:
            raise ValueError(
                f"Response marker '{marker}' not found in full_input. "
                f"Check chat template for {self.model_name}."
            )
        return idx + len(marker)

    def validate_sign_position(self, sign_char_offset: int, prefix_len: int,
                                full_input: str, wrong_sign: str) -> bool:
        """
        F1 contamination check: sign must be in response, not prompt.
        Returns True if valid, raises if contaminated.
        """
        # Primary check: offset must be past prefix
        assert sign_char_offset > prefix_len, (
            f"CONTAMINATION: sign_char_offset={sign_char_offset} <= "
            f"prefix_len={prefix_len}. Sign is inside the prompt."
        )
        # Secondary check: character at offset must match wrong_sign
        actual_char = full_input[sign_char_offset] if sign_char_offset < len(full_input) else "?"
        assert actual_char == wrong_sign, (
            f"Sign mismatch: full_input[{sign_char_offset}]='{actual_char}' "
            f"but wrong_sign='{wrong_sign}'"
        )
        return True

    def get_sign_token_ids(self, wrong_sign: str, sign_char_offset: int,
                           full_input: str) -> tuple[int, int]:
        """
        Return (wrong_sign_tok_id, correct_sign_tok_id).
        Handles space-prefix: if char before sign is a space, tokenize ' +'/'-'.
        Returns IDs for: wrong_sign_tok = what model wrote, correct_sign_tok = the other sign.
        """
        has_space = (sign_char_offset > 0 and full_input[sign_char_offset - 1] == " ")
        prefix = " " if has_space else ""

        plus_str  = prefix + "+"
        minus_str = prefix + "-"

        plus_ids  = self.tokenizer(plus_str,  add_special_tokens=False).input_ids
        minus_ids = self.tokenizer(minus_str, add_special_tokens=False).input_ids

        # Use the last token ID (in case tokenizer produces multiple tokens)
        plus_tok  = plus_ids[-1]
        minus_tok = minus_ids[-1]

        # Sanity: both must be different
        if plus_tok == minus_tok:
            raise ValueError(
                f"Tokenizer cannot distinguish '+' from '-' for {self.model_name}. "
                f"Both map to token {plus_tok}. Check tokenizer."
            )

        if wrong_sign == "+":
            return plus_tok, minus_tok    # written=+, opposite=-
        else:
            return minus_tok, plus_tok    # written=-, opposite=+

    def char_to_token_idx(self, char_offset: int, full_input: str) -> int:
        """
        Convert character offset in full_input to token index.
        Returns the token index such that full_input[char_offset] is inside that token.
        """
        enc = self.tokenizer(
            full_input,
            return_offsets_mapping=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        offsets = enc["offset_mapping"][0].tolist()
        for i, (start, end) in enumerate(offsets):
            if start <= char_offset < end:
                return i
        # If exact match fails, find nearest
        for i, (start, end) in enumerate(offsets):
            if start <= char_offset:
                last = i
        return last

    # ── Core computation (model-specific) ─────────────────────────────────────

    @abstractmethod
    def compute_logit_diff(self, h: torch.Tensor,
                           wrong_sign_tok: int, correct_sign_tok: int) -> float:
        """
        Project hidden state h through final LN + lm_head.
        Return logit[wrong_sign_tok] - logit[correct_sign_tok].
        Positive = model prefers the wrong sign (what it wrote).
        Negative = model prefers the correct sign.
        Implements all model-specific corrections (softcap, (1+w) norm, etc.)
        """

    @abstractmethod
    def get_layer_modules(self) -> list:
        """Return list of transformer layer modules (for hooking)."""

    @abstractmethod
    def get_mlp_module(self, layer_idx: int):
        """Return the MLP module of layer layer_idx."""

    @abstractmethod
    def get_attn_module(self, layer_idx: int):
        """Return the attention output projection module of layer layer_idx."""

    @abstractmethod
    def get_o_proj(self, layer_idx: int):
        """Return the o_proj Linear module of layer layer_idx."""

    @abstractmethod
    def get_embedding_output(self, full_ids: torch.Tensor,
                              probe_tok: int) -> torch.Tensor:
        """Return the token embedding at probe_tok position."""

    @abstractmethod
    def compute_eff_dir(self, wrong_sign_tok: int, correct_sign_tok: int,
                         h_final: torch.Tensor) -> torch.Tensor:
        """
        Compute the effective direction for DLA.
        Standard: (W_U[wrong_sign_tok] - W_U[correct_sign_tok]) * final_LN_scale * norm_weight
        Gemma-3 override: applies softcap Jacobian correction to each term.
        """

    # ── Tokenization ──────────────────────────────────────────────────────────

    def tokenize(self, full_input: str) -> torch.Tensor:
        """Tokenize full_input, return input_ids tensor on first GPU."""
        device = next(self.model.parameters()).device
        ids = self.tokenizer(full_input, return_tensors="pt",
                             add_special_tokens=False).input_ids
        return ids.to(device)
