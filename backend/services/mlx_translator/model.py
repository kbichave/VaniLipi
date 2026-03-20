"""
MLX encoder-decoder model for IndicTrans2.

Architecture (confirmed from the IndicTrans2 paper and model config):
  - Encoder: 18 layers, d_model=1024, ffn_dim=8192, 16 heads, pre-norm, GELU
  - Decoder: 18 layers, same dims, with cross-attention + causal self-attention
  - Separate encoder / decoder embedding tables (NOT shared vocab, unlike NLLB/M2M)
  - Sinusoidal positional embeddings (no learned)

The 200M distilled model uses fewer layers (confirmed: 12 enc + 12 dec) with the same
d_model/head dims — use IT2Config.from_model_config() to read actual values from the
downloaded config.json rather than hard-coding.

Design note on memory:
  mlx.nn.MultiHeadAttention follows (query, key, value) positional convention.
  KV cache for autoregressive decoding is handled externally in generate.py via
  concatenation (matching mlx-whisper's approach) rather than MLX's stateful cache API.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class IT2Config:
    encoder_layers: int = 18
    decoder_layers: int = 18
    d_model: int = 1024
    ffn_dim: int = 8192
    attention_heads: int = 16
    encoder_vocab_size: int = 256206   # from model config.json
    decoder_vocab_size: int = 256206
    max_position_embeddings: int = 256
    dropout: float = 0.0              # disabled for inference
    pad_token_id: int = 1
    bos_token_id: int = 2
    eos_token_id: int = 2

    @classmethod
    def from_model_config(cls, config_path: str | Path) -> "IT2Config":
        """Load config from the HuggingFace model's config.json."""
        with open(config_path, "r") as f:
            cfg = json.load(f)

        return cls(
            encoder_layers=cfg.get("encoder_layers", 18),
            decoder_layers=cfg.get("decoder_layers", 18),
            d_model=cfg.get("encoder_embed_dim", cfg.get("d_model", 1024)),
            ffn_dim=cfg.get("encoder_ffn_dim", 8192),
            attention_heads=cfg.get("encoder_attention_heads", 16),
            encoder_vocab_size=cfg.get("encoder_vocab_size", cfg.get("vocab_size", 256206)),
            decoder_vocab_size=cfg.get("decoder_vocab_size", cfg.get("vocab_size", 256206)),
            max_position_embeddings=cfg.get("max_source_positions", cfg.get("max_position_embeddings", 256)),
            pad_token_id=cfg.get("pad_token_id", 1),
            bos_token_id=cfg.get("bos_token_id", 2),
            eos_token_id=cfg.get("eos_token_id", 2),
        )


# ---------------------------------------------------------------------------
# Positional embeddings (fairseq convention, matching HF IndicTrans2)
# ---------------------------------------------------------------------------

def _build_sinusoidal_table(num_positions: int, d_model: int, padding_idx: int = 1) -> mx.array:
    """
    Build sinusoidal PE table matching fairseq/HF IndicTrans2 convention.

    Key differences from vanilla Vaswani PE:
      - Div term uses half_dim - 1 as divisor (fairseq convention)
      - Layout is [sin_all, cos_all] concatenated, NOT interleaved
      - Row at padding_idx is zeroed out

    Shape: (num_positions, d_model)
    """
    half_dim = d_model // 2
    # fairseq: log(10000) / (half_dim - 1)
    emb = math.log(10000.0) / (half_dim - 1)
    div_term = mx.exp(mx.arange(half_dim, dtype=mx.float32) * -emb)  # (half_dim,)
    position = mx.arange(num_positions, dtype=mx.float32)[:, None]    # (N, 1)

    sin_vals = mx.sin(position * div_term[None, :])  # (N, half_dim)
    cos_vals = mx.cos(position * div_term[None, :])  # (N, half_dim)

    # Concatenated layout: [sin_0..sin_{d/2-1}, cos_0..cos_{d/2-1}]
    pe = mx.concatenate([sin_vals, cos_vals], axis=1)  # (N, d_model)

    # Zero out the padding_idx row
    zero_mask = (mx.arange(num_positions) != padding_idx).astype(mx.float32)[:, None]
    pe = pe * zero_mask

    return pe


def _create_position_ids_from_input_ids(input_ids: mx.array, padding_idx: int) -> mx.array:
    """
    Create position IDs from input_ids, matching HF IndicTrans2's convention.

    Non-padding tokens get sequential positions starting from padding_idx + 1.
    Padding tokens get padding_idx (which maps to a zero PE row).

    Example (padding_idx=1):
      input_ids:   [5, 3, 7, 1, 1]
      position_ids: [2, 3, 4, 1, 1]
    """
    mask = (input_ids != padding_idx).astype(mx.int32)
    incremental_indices = mx.cumsum(mask, axis=1) * mask
    return incremental_indices + padding_idx


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class TransformerEncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer (used by IndicTrans2)."""

    def __init__(self, config: IT2Config) -> None:
        super().__init__()
        self.self_attn = nn.MultiHeadAttention(config.d_model, config.attention_heads, bias=True)
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.ffn_dim)
        self.fc2 = nn.Linear(config.ffn_dim, config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        # Pre-norm self-attention
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, x, x, mask=mask)
        if isinstance(x, tuple):
            x = x[0]  # MultiHeadAttention may return (output, weights)
        x = residual + x

        # Pre-norm FFN
        residual = x
        x = self.final_layer_norm(x)
        x = nn.gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x
        return x


@dataclass
class DecoderLayerCache:
    """KV cache for one decoder layer. Mutated in-place during generation."""
    self_k: Optional[mx.array] = None   # (batch, past_seq, d_model)
    self_v: Optional[mx.array] = None
    cross_k: Optional[mx.array] = None  # (batch, enc_seq, d_model) — static
    cross_v: Optional[mx.array] = None


class TransformerDecoderLayer(nn.Module):
    """Pre-norm transformer decoder layer with cross-attention."""

    def __init__(self, config: IT2Config) -> None:
        super().__init__()
        self.self_attn = nn.MultiHeadAttention(config.d_model, config.attention_heads, bias=True)
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.encoder_attn = nn.MultiHeadAttention(config.d_model, config.attention_heads, bias=True)
        self.encoder_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.ffn_dim)
        self.fc2 = nn.Linear(config.ffn_dim, config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model)

    def __call__(
        self,
        x: mx.array,
        encoder_out: mx.array,
        self_mask: Optional[mx.array] = None,
        cross_mask: Optional[mx.array] = None,
    ) -> mx.array:
        # Pre-norm causal self-attention
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, x, x, mask=self_mask)
        if isinstance(x, tuple):
            x = x[0]
        x = residual + x

        # Pre-norm cross-attention
        residual = x
        x = self.encoder_attn_layer_norm(x)
        x = self.encoder_attn(x, encoder_out, encoder_out, mask=cross_mask)
        if isinstance(x, tuple):
            x = x[0]
        x = residual + x

        # Pre-norm FFN
        residual = x
        x = self.final_layer_norm(x)
        x = nn.gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x
        return x

    def cached_forward(
        self,
        x: mx.array,
        encoder_out: mx.array,
        cache: DecoderLayerCache,
        cross_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Forward pass with KV cache — only processes the last token position.

        On first call (cache empty): processes full sequence, populates cache.
        On subsequent calls: processes one new token, appends to cache.
        """
        # Pre-norm self-attention with cache
        residual = x
        normed = self.self_attn_layer_norm(x)

        # Project current token(s) to K/V and append to cache
        sa = self.self_attn
        new_k = normed @ sa.key_proj.weight.T + sa.key_proj.bias
        new_v = normed @ sa.value_proj.weight.T + sa.value_proj.bias
        q = normed @ sa.query_proj.weight.T + sa.query_proj.bias

        if cache.self_k is not None:
            cache.self_k = mx.concatenate([cache.self_k, new_k], axis=1)
            cache.self_v = mx.concatenate([cache.self_v, new_v], axis=1)
        else:
            cache.self_k = new_k
            cache.self_v = new_v

        # Reshape for multi-head attention: (batch, heads, seq, head_dim)
        batch = q.shape[0]
        num_heads = sa.num_heads
        head_dim = q.shape[-1] // num_heads

        q = q.reshape(batch, -1, num_heads, head_dim).transpose(0, 2, 1, 3)
        k = cache.self_k.reshape(batch, -1, num_heads, head_dim).transpose(0, 2, 1, 3)
        v = cache.self_v.reshape(batch, -1, num_heads, head_dim).transpose(0, 2, 1, 3)

        # Scaled dot-product attention (no mask needed — causal is implicit
        # since we only query the last position against all cached positions)
        scale = math.sqrt(head_dim)
        scores = (q @ k.transpose(0, 1, 3, 2)) / scale
        weights = mx.softmax(scores, axis=-1)
        attn_out = (weights @ v).transpose(0, 2, 1, 3).reshape(batch, -1, num_heads * head_dim)
        attn_out = attn_out @ sa.out_proj.weight.T + sa.out_proj.bias
        x = residual + attn_out

        # Pre-norm cross-attention with cache
        residual = x
        normed = self.encoder_attn_layer_norm(x)

        ca = self.encoder_attn
        cq = normed @ ca.query_proj.weight.T + ca.query_proj.bias

        # Cross K/V are static — compute once and cache
        if cache.cross_k is None:
            cache.cross_k = encoder_out @ ca.key_proj.weight.T + ca.key_proj.bias
            cache.cross_v = encoder_out @ ca.value_proj.weight.T + ca.value_proj.bias

        cq = cq.reshape(batch, -1, num_heads, head_dim).transpose(0, 2, 1, 3)
        ck = cache.cross_k.reshape(batch, -1, num_heads, head_dim).transpose(0, 2, 1, 3)
        cv = cache.cross_v.reshape(batch, -1, num_heads, head_dim).transpose(0, 2, 1, 3)

        scores = (cq @ ck.transpose(0, 1, 3, 2)) / scale
        if cross_mask is not None:
            scores = scores + cross_mask
        weights = mx.softmax(scores, axis=-1)
        cross_out = (weights @ cv).transpose(0, 2, 1, 3).reshape(batch, -1, num_heads * head_dim)
        cross_out = cross_out @ ca.out_proj.weight.T + ca.out_proj.bias
        x = residual + cross_out

        # Pre-norm FFN
        residual = x
        x = self.final_layer_norm(x)
        x = nn.gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x
        return x


# ---------------------------------------------------------------------------
# Full encoder / decoder
# ---------------------------------------------------------------------------

class IT2Encoder(nn.Module):
    def __init__(self, config: IT2Config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.encoder_vocab_size, config.d_model)
        self.embed_scale = math.sqrt(config.d_model)
        self.layers = [TransformerEncoderLayer(config) for _ in range(config.encoder_layers)]
        self.layer_norm = nn.LayerNorm(config.d_model)
        self._pad_token_id = config.pad_token_id
        # Pre-build PE table: max_positions + padding_idx + 1 rows
        self._pe_table = _build_sinusoidal_table(
            config.max_position_embeddings + config.pad_token_id + 1,
            config.d_model,
            padding_idx=config.pad_token_id,
        )

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        x = self.embed_tokens(input_ids) * self.embed_scale

        # Padding-aware position IDs (non-pad tokens start at padding_idx + 1)
        position_ids = _create_position_ids_from_input_ids(input_ids, self._pad_token_id)
        x = x + self._pe_table[position_ids]

        # Convert padding mask (1=keep, 0=pad) to additive attention mask (-inf for pad)
        mask = _padding_mask_to_additive(attention_mask) if attention_mask is not None else None

        for layer in self.layers:
            x = layer(x, mask=mask)

        return self.layer_norm(x)


class IT2Decoder(nn.Module):
    def __init__(self, config: IT2Config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.decoder_vocab_size, config.d_model)
        self.embed_scale = math.sqrt(config.d_model)
        self.layers = [TransformerDecoderLayer(config) for _ in range(config.decoder_layers)]
        self.layer_norm = nn.LayerNorm(config.d_model)
        self._pad_token_id = config.pad_token_id
        # Pre-build PE table (same convention as encoder)
        self._pe_table = _build_sinusoidal_table(
            config.max_position_embeddings + config.pad_token_id + 1,
            config.d_model,
            padding_idx=config.pad_token_id,
        )

    def __call__(
        self,
        input_ids: mx.array,
        encoder_hidden_states: mx.array,
        encoder_attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        seq_len = input_ids.shape[1]
        x = self.embed_tokens(input_ids) * self.embed_scale

        # During generation, decoder input has no padding — positions start at padding_idx + 1
        position_ids = _create_position_ids_from_input_ids(input_ids, self._pad_token_id)
        x = x + self._pe_table[position_ids]

        # Causal mask for decoder self-attention
        causal_mask = _causal_mask(seq_len)
        cross_mask = (
            _padding_mask_to_additive(encoder_attention_mask)
            if encoder_attention_mask is not None
            else None
        )

        for layer in self.layers:
            x = layer(
                x,
                encoder_hidden_states,
                self_mask=causal_mask,
                cross_mask=cross_mask,
            )

        return self.layer_norm(x)

    def cached_forward(
        self,
        input_ids: mx.array,
        encoder_hidden_states: mx.array,
        cache: list[DecoderLayerCache],
        past_length: int,
        encoder_attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Cached decoder forward — processes only the new token(s).

        Args:
            input_ids:              New token IDs only, shape (batch, new_tokens).
            encoder_hidden_states:  Encoder output (unchanged between steps).
            cache:                  Per-layer KV cache (mutated in-place).
            past_length:            Number of tokens already in the cache.
            encoder_attention_mask: Encoder padding mask.
        """
        x = self.embed_tokens(input_ids) * self.embed_scale

        # Position IDs for new tokens: continue from past_length
        # Positions start at padding_idx + 1, so offset accordingly
        new_len = input_ids.shape[1]
        start_pos = past_length + self._pad_token_id + 1
        position_ids = mx.arange(start_pos, start_pos + new_len)[None, :]  # (1, new_len)
        position_ids = mx.broadcast_to(position_ids, input_ids.shape)
        x = x + self._pe_table[position_ids]

        cross_mask = (
            _padding_mask_to_additive(encoder_attention_mask)
            if encoder_attention_mask is not None
            else None
        )

        for layer, layer_cache in zip(self.layers, cache):
            x = layer.cached_forward(
                x, encoder_hidden_states, layer_cache, cross_mask=cross_mask,
            )

        return self.layer_norm(x)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class IndicTrans2(nn.Module):
    """
    Full IndicTrans2 encoder-decoder model for MLX inference.

    The 1B model does NOT share decoder embeddings with lm_head
    (share_decoder_input_output_embed=false in config). A separate
    lm_head Linear projection is used.
    """

    def __init__(self, config: IT2Config) -> None:
        super().__init__()
        self.config = config
        self.encoder = IT2Encoder(config)
        self.decoder = IT2Decoder(config)
        # Separate output projection (not tied to decoder embeddings)
        self.lm_head = nn.Linear(config.d_model, config.decoder_vocab_size, bias=False)

    def encode(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        return self.encoder(input_ids, attention_mask=attention_mask)

    def decode_step(
        self,
        decoder_input_ids: mx.array,
        encoder_hidden_states: mx.array,
        encoder_attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """
        One full decoder forward pass (no KV cache).
        Returns logits of shape (batch, seq_len, vocab_size).
        """
        hidden = self.decoder(
            decoder_input_ids,
            encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        return self.lm_head(hidden)

    def decode_step_cached(
        self,
        decoder_input_ids: mx.array,
        encoder_hidden_states: mx.array,
        cache: list[DecoderLayerCache],
        past_length: int,
        encoder_attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Cached decoder forward — only computes the new token(s).
        Returns logits for the new positions only: (batch, new_tokens, vocab_size).
        """
        hidden = self.decoder.cached_forward(
            decoder_input_ids,
            encoder_hidden_states,
            cache,
            past_length,
            encoder_attention_mask=encoder_attention_mask,
        )
        return self.lm_head(hidden)

    def make_cache(self) -> list[DecoderLayerCache]:
        """Create an empty KV cache (one entry per decoder layer)."""
        return [DecoderLayerCache() for _ in range(self.config.decoder_layers)]

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        decoder_input_ids: Optional[mx.array] = None,
    ) -> mx.array:
        enc_out = self.encode(input_ids, attention_mask=attention_mask)
        if decoder_input_ids is None:
            # Return encoder output only (useful for inspect/debug)
            return enc_out
        return self.decode_step(decoder_input_ids, enc_out, encoder_attention_mask=attention_mask)


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def _causal_mask(seq_len: int) -> mx.array:
    """
    Upper-triangular additive causal mask.
    Positions that should be masked get -1e9; others get 0.
    Shape: (1, 1, seq_len, seq_len)
    """
    mask = mx.triu(mx.ones((seq_len, seq_len)), k=1)
    return (-1e9 * mask)[None, None, :, :]


def _padding_mask_to_additive(mask: mx.array) -> mx.array:
    """
    Convert a boolean/integer padding mask (1=keep, 0=pad) to an additive
    attention mask compatible with MLX MultiHeadAttention.
    Shape in:  (batch, seq_len)
    Shape out: (batch, 1, 1, seq_len)
    """
    # 0 → -1e9 (masked), 1 → 0 (not masked)
    additive = (1.0 - mask.astype(mx.float32)) * -1e9
    return additive[:, None, None, :]
