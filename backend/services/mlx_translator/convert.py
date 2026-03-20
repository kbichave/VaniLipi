"""
Weight conversion: PyTorch (HuggingFace safetensors) → MLX safetensors.

The IndicTrans2 HF model uses these key prefixes:
  model.encoder.embed_tokens.weight
  model.encoder.layers.N.self_attn.{q,k,v,out}_proj.{weight,bias}
  model.encoder.layers.N.self_attn_layer_norm.{weight,bias}
  model.encoder.layers.N.fc{1,2}.{weight,bias}
  model.encoder.layers.N.final_layer_norm.{weight,bias}
  model.encoder.layer_norm.{weight,bias}
  model.decoder.* (same structure)

Our MLX model uses:
  encoder.embed_tokens.weight
  encoder.layers.N.self_attn.{query,key,value,out}_proj.{weight,bias}
  encoder.layers.N.self_attn_layer_norm.{weight,bias}
  encoder.layers.N.fc{1,2}.{weight,bias}
  encoder.layers.N.final_layer_norm.{weight,bias}
  encoder.layer_norm.{weight,bias}
  decoder.* (same structure)

MLX MultiHeadAttention expects separate query_proj / key_proj / value_proj / out_proj
with the same weight shapes as the HF q_proj / k_proj / v_proj / o_proj.

Usage (run from project root after activating venv):
    python -m backend.services.mlx_translator.convert \\
        --hf-model ai4bharat/indictrans2-indic-en-1B \\
        --output ~/.cache/vanilipi/mlx_indictrans2_1b/weights.safetensors

The script downloads the HF model on first run (cached in ~/.cache/huggingface/).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import mlx.core as mx
import numpy as np
mlx_save = mx.save_safetensors

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Key remapping
# ---------------------------------------------------------------------------

def remap_key(hf_key: str) -> str | None:
    """
    Map a HuggingFace model key to our MLX model key.
    Returns None for keys we should drop (e.g. lm_head — it's tied to embed_tokens).

    Mapping rules:
      model.encoder.* → encoder.*
      model.decoder.* → decoder.*
      .q_proj         → .self_attn.query_proj  (inside self_attn)
      .k_proj         → .self_attn.key_proj
      .v_proj         → .self_attn.value_proj
      .out_proj       → .self_attn.out_proj
      encoder_attn.{q,k,v,out}_proj → encoder_attn.{query,key,value,out}_proj
      lm_head.*       → None (tied embedding, skip)
    """
    # Drop lm_head (it's weight-tied to decoder.embed_tokens)
    if hf_key.startswith("lm_head"):
        return None

    # Strip "model." prefix
    key = hf_key
    if key.startswith("model."):
        key = key[len("model."):]

    # Rename attention projection keys inside self_attn
    # Pattern: (encoder|decoder).layers.N.self_attn.{q,k,v,out}_proj
    for side in ("encoder", "decoder"):
        for layer_type in ("self_attn", "encoder_attn"):
            prefix = f"{side}.layers."
            if key.startswith(prefix) and f".{layer_type}.q_proj" in key:
                key = key.replace(f".{layer_type}.q_proj", f".{layer_type}.query_proj")
            if key.startswith(prefix) and f".{layer_type}.k_proj" in key:
                key = key.replace(f".{layer_type}.k_proj", f".{layer_type}.key_proj")
            if key.startswith(prefix) and f".{layer_type}.v_proj" in key:
                key = key.replace(f".{layer_type}.v_proj", f".{layer_type}.value_proj")
            if key.startswith(prefix) and f".{layer_type}.out_proj" in key:
                key = key.replace(f".{layer_type}.out_proj", f".{layer_type}.out_proj")

    return key


def convert(hf_model_id: str, output_path: str | Path) -> None:
    """
    Download (if needed) and convert the HF model weights to MLX format.

    Args:
        hf_model_id:  HuggingFace repo ID, e.g. "ai4bharat/indictrans2-indic-en-1B"
        output_path:  Where to save the MLX safetensors file.
    """
    from transformers import AutoModelForSeq2SeqLM  # type: ignore[import]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading PyTorch model: %s (this may take a minute)", hf_model_id)
    pt_model = AutoModelForSeq2SeqLM.from_pretrained(hf_model_id, trust_remote_code=True)
    pt_state = pt_model.state_dict()
    logger.info("Loaded %d PyTorch weight tensors.", len(pt_state))

    mlx_weights: dict[str, mx.array] = {}
    skipped = []

    for hf_key, tensor in pt_state.items():
        mlx_key = remap_key(hf_key)
        if mlx_key is None:
            skipped.append(hf_key)
            continue
        # Convert to float16 for MLX (matches the PyTorch fp16 inference setup)
        arr = mx.array(tensor.detach().float().numpy()).astype(mx.float16)
        mlx_weights[mlx_key] = arr

    logger.info("Converted %d tensors, skipped %d (lm_head/tied).", len(mlx_weights), len(skipped))

    mlx_save(str(output_path), mlx_weights)
    logger.info("Saved MLX weights to %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Convert IndicTrans2 HF → MLX weights")
    parser.add_argument(
        "--hf-model",
        default="ai4bharat/indictrans2-indic-en-1B",
        help="HuggingFace model ID (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default=str(
            Path.home() / ".cache" / "vanilipi" / "mlx_indictrans2_1b" / "weights.safetensors"
        ),
        help="Output path for MLX weights (default: %(default)s)",
    )
    args = parser.parse_args()
    convert(args.hf_model, args.output)


if __name__ == "__main__":
    main()
