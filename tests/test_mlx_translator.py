"""
Tests for Phase 2.5: MLX IndicTrans2 model and conversion utilities.
All MLX operations are mocked — no GPU or model downloads required.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Model architecture tests
# ---------------------------------------------------------------------------

class TestIT2Config:
    def test_defaults(self):
        from backend.services.mlx_translator.model import IT2Config

        cfg = IT2Config()
        assert cfg.encoder_layers == 18
        assert cfg.decoder_layers == 18
        assert cfg.d_model == 1024
        assert cfg.ffn_dim == 8192
        assert cfg.attention_heads == 16

    def test_from_model_config_reads_json(self, tmp_path):
        import json
        from backend.services.mlx_translator.model import IT2Config

        config_data = {
            "encoder_layers": 12,
            "decoder_layers": 12,
            "d_model": 512,
            "encoder_ffn_dim": 4096,
            "encoder_attention_heads": 8,
            "vocab_size": 64000,
            "max_position_embeddings": 512,
            "pad_token_id": 1,
            "bos_token_id": 2,
            "eos_token_id": 2,
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        cfg = IT2Config.from_model_config(config_file)
        assert cfg.encoder_layers == 12
        assert cfg.d_model == 512
        assert cfg.attention_heads == 8
        assert cfg.encoder_vocab_size == 64000

    def test_head_divides_d_model(self):
        from backend.services.mlx_translator.model import IT2Config

        cfg = IT2Config()
        assert cfg.d_model % cfg.attention_heads == 0, (
            f"d_model={cfg.d_model} must be divisible by attention_heads={cfg.attention_heads}"
        )


class TestSinusoidalPE:
    def test_output_shape(self):
        from backend.services.mlx_translator.model import sinusoidal_positional_embeddings
        import mlx.core as mx

        pe = sinusoidal_positional_embeddings(10, 64)
        assert pe.shape == (10, 64)

    def test_different_positions_differ(self):
        from backend.services.mlx_translator.model import sinusoidal_positional_embeddings
        import mlx.core as mx

        pe = sinusoidal_positional_embeddings(5, 16)
        # No two position vectors should be identical
        for i in range(5):
            for j in range(i + 1, 5):
                diff = mx.abs(pe[i] - pe[j]).tolist()
                assert any(d > 1e-6 for d in diff), f"PE rows {i} and {j} are identical"

    def test_consistent_dimensions(self):
        from backend.services.mlx_translator.model import sinusoidal_positional_embeddings

        for d in (32, 64, 128, 256, 512, 1024):
            pe = sinusoidal_positional_embeddings(20, d)
            assert pe.shape == (20, d)


class TestMaskHelpers:
    def test_causal_mask_shape(self):
        from backend.services.mlx_translator.model import _causal_mask

        mask = _causal_mask(5)
        assert mask.shape == (1, 1, 5, 5)

    def test_causal_mask_upper_triangle_is_negative(self):
        from backend.services.mlx_translator.model import _causal_mask

        mask = _causal_mask(4)
        m = mask[0, 0].tolist()
        # Upper triangle (i < j) should be very negative
        for i in range(4):
            for j in range(4):
                if j > i:
                    assert m[i][j] < -1e8, f"mask[{i},{j}] should be masked but got {m[i][j]}"
                else:
                    assert m[i][j] == 0.0, f"mask[{i},{j}] should be 0 but got {m[i][j]}"

    def test_padding_mask_converts_zeros_to_negative(self):
        from backend.services.mlx_translator.model import _padding_mask_to_additive
        import mlx.core as mx

        # 1=keep, 0=pad
        mask = mx.array([[1, 1, 0]], dtype=mx.float32)  # batch=1, seq=3
        additive = _padding_mask_to_additive(mask)
        assert additive.shape == (1, 1, 1, 3)
        vals = additive[0, 0, 0].tolist()
        assert vals[0] == 0.0  # keep
        assert vals[1] == 0.0  # keep
        assert vals[2] < -1e8   # pad


class TestEncoderDecoderShapes:
    """Test model forward shapes without loading real weights."""

    def _build_model(self):
        from backend.services.mlx_translator.model import IT2Config, IndicTrans2
        import mlx.core as mx

        # Tiny config for fast tests
        cfg = IT2Config(
            encoder_layers=1,
            decoder_layers=1,
            d_model=32,
            ffn_dim=64,
            attention_heads=4,
            encoder_vocab_size=100,
            decoder_vocab_size=100,
            max_position_embeddings=32,
        )
        return IndicTrans2(cfg), cfg

    def test_encoder_output_shape(self):
        import mlx.core as mx

        model, cfg = self._build_model()
        batch, seq = 2, 5
        input_ids = mx.zeros((batch, seq), dtype=mx.int32)
        mask = mx.ones((batch, seq), dtype=mx.float32)

        enc_out = model.encode(input_ids, attention_mask=mask)
        mx.eval(enc_out)
        assert enc_out.shape == (batch, seq, cfg.d_model)

    def test_decode_step_output_shape(self):
        import mlx.core as mx

        model, cfg = self._build_model()
        batch, enc_seq, dec_seq = 2, 5, 3
        enc_hidden = mx.zeros((batch, enc_seq, cfg.d_model))
        dec_ids = mx.zeros((batch, dec_seq), dtype=mx.int32)

        logits = model.decode_step(dec_ids, enc_hidden)
        mx.eval(logits)
        assert logits.shape == (batch, dec_seq, cfg.decoder_vocab_size)


# ---------------------------------------------------------------------------
# Key remapping tests
# ---------------------------------------------------------------------------

class TestRemapKey:
    def test_strips_model_prefix(self):
        from backend.services.mlx_translator.convert import remap_key

        assert remap_key("model.encoder.layer_norm.weight") == "encoder.layer_norm.weight"

    def test_renames_q_proj(self):
        from backend.services.mlx_translator.convert import remap_key

        result = remap_key("model.encoder.layers.0.self_attn.q_proj.weight")
        assert "query_proj" in result

    def test_renames_k_proj(self):
        from backend.services.mlx_translator.convert import remap_key

        result = remap_key("model.encoder.layers.0.self_attn.k_proj.weight")
        assert "key_proj" in result

    def test_renames_v_proj(self):
        from backend.services.mlx_translator.convert import remap_key

        result = remap_key("model.encoder.layers.0.self_attn.v_proj.weight")
        assert "value_proj" in result

    def test_drops_lm_head(self):
        from backend.services.mlx_translator.convert import remap_key

        assert remap_key("lm_head.weight") is None

    def test_decoder_cross_attention_remapped(self):
        from backend.services.mlx_translator.convert import remap_key

        result = remap_key("model.decoder.layers.0.encoder_attn.q_proj.weight")
        assert result is not None
        assert "query_proj" in result

    def test_embed_tokens_passthrough(self):
        from backend.services.mlx_translator.convert import remap_key

        result = remap_key("model.encoder.embed_tokens.weight")
        assert result == "encoder.embed_tokens.weight"


# ---------------------------------------------------------------------------
# Flatten params
# ---------------------------------------------------------------------------

class TestFlattenParams:
    def test_flat_dict_unchanged(self):
        import mlx.core as mx
        from backend.services.mlx_translator.quantize import _flatten_params

        w = mx.zeros((3, 3))
        result = _flatten_params({"weight": w})
        assert "weight" in result

    def test_nested_dict_flattened(self):
        import mlx.core as mx
        from backend.services.mlx_translator.quantize import _flatten_params

        w = mx.zeros((3, 3))
        result = _flatten_params({"encoder": {"layer_norm": {"weight": w}}})
        assert "encoder.layer_norm.weight" in result

    def test_list_indexed(self):
        import mlx.core as mx
        from backend.services.mlx_translator.quantize import _flatten_params

        w = mx.zeros((2, 2))
        result = _flatten_params({"layers": [{"weight": w}, {"weight": w}]})
        assert "layers.0.weight" in result
        assert "layers.1.weight" in result


# ---------------------------------------------------------------------------
# Beam search
# ---------------------------------------------------------------------------

class TestBeamSearch:
    def _make_model_mock(self, vocab_size: int = 50, always_token: int = 5):
        """Create a fake model that always returns high score for `always_token`."""
        import mlx.core as mx

        model = MagicMock()

        def fake_decode_step(decoder_ids, enc_hidden, encoder_attention_mask=None):
            import numpy as np
            batch = decoder_ids.shape[0]
            seq = decoder_ids.shape[1]
            logits_np = np.zeros((batch, seq, vocab_size), dtype=np.float32)
            # Make always_token very likely at every position
            logits_np[:, :, always_token] = 20.0
            return mx.array(logits_np)

        model.decode_step = fake_decode_step
        return model

    def test_returns_list_of_lists(self):
        import mlx.core as mx
        from backend.services.mlx_translator.generate import beam_search

        model = self._make_model_mock(vocab_size=50, always_token=10)
        enc = mx.zeros((2, 5, 32))
        mask = mx.ones((2, 5), dtype=mx.float32)

        results = beam_search(
            model, enc, mask,
            bos_token_id=1, eos_token_id=2, pad_token_id=0,
            max_length=10, num_beams=3,
        )
        assert isinstance(results, list)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, list)

    def test_stops_at_max_length(self):
        import mlx.core as mx
        from backend.services.mlx_translator.generate import beam_search

        # EOS token is 99 (never emitted), so we should hit max_length
        model = self._make_model_mock(vocab_size=50, always_token=7)
        enc = mx.zeros((1, 4, 32))
        mask = mx.ones((1, 4), dtype=mx.float32)

        results = beam_search(
            model, enc, mask,
            bos_token_id=1, eos_token_id=99, pad_token_id=0,
            max_length=8, num_beams=2,
        )
        assert len(results[0]) <= 8


# ---------------------------------------------------------------------------
# Validate test sentences list
# ---------------------------------------------------------------------------

class TestValidateSentences:
    def test_50_sentences(self):
        from backend.services.mlx_translator.validate import TEST_SENTENCES

        assert len(TEST_SENTENCES) == 50

    def test_all_non_empty(self):
        from backend.services.mlx_translator.validate import TEST_SENTENCES

        for s in TEST_SENTENCES:
            assert s.strip(), f"Empty sentence in TEST_SENTENCES: {s!r}"
