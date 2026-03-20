"""
Beam search decoding for MLX IndicTrans2 with KV cache.

Design based on mlx-whisper's BeamSearchDecoder (MIT licensed), adapted for:
  - IndicTrans2's separate decoder vocabulary
  - Batch inference (multiple sentences per call)
  - IndicTrans2's BOS token (lang tag) + EOS token
  - KV cache for O(n) decoding instead of O(n²) per step

The KV cache stores projected key/value tensors for self-attention (growing)
and cross-attention (static after first step). On each step, only the new
token is fed through the decoder, and the cache is updated in-place.

When beams are reordered (a beam from position j becomes position i), the
cache must be reindexed to match the new beam layout.

Failure modes handled:
  - Beam candidates that exceed max_length are finalized early
  - If all beams produce only EOS, stop early (length penalty still applied)
  - Empty hypothesis after generation → fallback to beam 0 output
"""
from __future__ import annotations

import logging
from typing import Optional

import mlx.core as mx

from backend.services.mlx_translator.model import DecoderLayerCache

logger = logging.getLogger(__name__)


def _advance_beams(
    batch_size: int,
    num_beams: int,
    vocab_size: int,
    top_scores: mx.array,
    top_indices: mx.array,
    beam_tokens: list[list[int]],
    beam_scores: mx.array,
    done: list[list[bool]],
    finished: list[list[tuple[float, list[int]]]],
    eos_token_id: int,
    length_penalty: float,
    past_length: int,
    cache: list[DecoderLayerCache],
) -> mx.array:
    """
    Advance beam search by one step: select top candidates, handle EOS,
    reorder cache for beam reassignment.

    Mutates beam_tokens, beam_scores_list, done, finished, and cache in-place.
    Returns updated beam_scores.
    """
    total_beams = batch_size * num_beams

    # Decode candidates per batch item
    new_beam_tokens: list[list[int]] = [[] for _ in range(total_beams)]
    new_scores: list[float] = [-1e9] * total_beams
    # Map: new beam global index → old beam global index (for cache reordering)
    reorder_map: list[int] = list(range(total_beams))

    for b in range(batch_size):
        added = 0
        for i in range(2 * num_beams):
            if added >= num_beams:
                break

            flat_idx = int(top_indices[b, i])
            beam_idx = flat_idx // vocab_size
            token_id = flat_idx % vocab_size
            score = float(top_scores[b, i])

            old_global = b * num_beams + beam_idx
            new_global = b * num_beams + added

            if done[b][beam_idx]:
                continue

            if token_id == eos_token_id:
                seq_len = len(beam_tokens[old_global]) + 1  # +1 for this EOS
                norm_score = score / (seq_len ** length_penalty)
                finished[b].append((norm_score, list(beam_tokens[old_global])))
                done[b][added] = True
                # Dead beam — keep tokens but mark score dead
                new_beam_tokens[new_global] = list(beam_tokens[old_global]) + [eos_token_id]
                new_scores[new_global] = -1e9
            else:
                new_beam_tokens[new_global] = list(beam_tokens[old_global]) + [token_id]
                new_scores[new_global] = score
                done[b][added] = done[b][beam_idx] if added != beam_idx else done[b][added]

            reorder_map[new_global] = old_global
            added += 1

        # Fill remaining beam slots if we ran out of live candidates
        while added < num_beams:
            new_global = b * num_beams + added
            new_beam_tokens[new_global] = []
            new_scores[new_global] = -1e9
            done[b][added] = True
            added += 1

    # Reorder cache if beams changed positions
    needs_reorder = any(reorder_map[i] != i for i in range(total_beams))
    if needs_reorder:
        idx = mx.array(reorder_map)
        for lc in cache:
            if lc.self_k is not None:
                lc.self_k = lc.self_k[idx]
                lc.self_v = lc.self_v[idx]
            if lc.cross_k is not None:
                lc.cross_k = lc.cross_k[idx]
                lc.cross_v = lc.cross_v[idx]

    # Update beam state
    for i in range(total_beams):
        beam_tokens[i] = new_beam_tokens[i]
    beam_scores_new = mx.array(new_scores, dtype=mx.float32)
    # Mutate the mutable container that the caller passed — we can't reassign
    # the caller's local variable, so we return the new scores.
    # The caller should use: beam_scores = _advance_beams(...)
    return beam_scores_new


def beam_search(
    model,
    encoder_hidden_states: mx.array,
    encoder_attention_mask: Optional[mx.array],
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    max_length: int = 256,
    num_beams: int = 5,
    length_penalty: float = 1.0,
    early_stopping: bool = True,
) -> list[list[int]]:
    """
    Beam search decoding for a batch of encoder hidden states.

    Uses KV cache: each step feeds only the newest token through the decoder.
    Self-attention K/V grow by one token per step; cross-attention K/V are
    computed once and reused.

    Args:
        model:                  IndicTrans2 MLX model (must have make_cache/decode_step_cached).
        encoder_hidden_states:  Shape (batch, enc_seq, d_model).
        encoder_attention_mask: Shape (batch, enc_seq) or None.
        bos_token_id:           First decoder token (language tag id).
        eos_token_id:           Stop token.
        pad_token_id:           Padding token id.
        max_length:             Max tokens to generate (including BOS).
        num_beams:              Number of beams per input.
        length_penalty:         >1.0 favors longer sequences.
        early_stopping:         Stop when all beams have produced EOS.

    Returns:
        List of token id lists (one per input in the batch), excluding BOS.
    """
    batch_size = encoder_hidden_states.shape[0]
    total_beams = batch_size * num_beams

    # Expand encoder states for beam search: (batch*beams, enc_seq, d_model)
    enc_hidden = mx.repeat(encoder_hidden_states, num_beams, axis=0)
    enc_mask = (
        mx.repeat(encoder_attention_mask, num_beams, axis=0)
        if encoder_attention_mask is not None
        else None
    )

    # Initialize KV cache (one per decoder layer, shared across all beams)
    cache = model.make_cache()

    # Step 0: feed BOS token through the full cached path to populate cache
    bos_ids = mx.full((total_beams, 1), bos_token_id, dtype=mx.int32)
    logits = model.decode_step_cached(
        bos_ids, enc_hidden, cache, past_length=0, encoder_attention_mask=enc_mask,
    )
    mx.eval(logits)
    _eval_cache(cache)

    next_token_logits = logits[:, -1, :]
    log_probs = _log_softmax(next_token_logits)
    vocab_size = log_probs.shape[-1]

    # Beam scores: only beam 0 per batch item is active initially
    beam_scores = mx.full((total_beams,), -1e9, dtype=mx.float32)
    for b in range(batch_size):
        beam_scores = _set_index(beam_scores, b * num_beams, 0.0)

    beam_tokens: list[list[int]] = [[] for _ in range(total_beams)]
    done = [[False] * num_beams for _ in range(batch_size)]
    finished: list[list[tuple[float, list[int]]]] = [[] for _ in range(batch_size)]

    past_length = 1  # BOS already in cache

    # Process step 1 (from BOS logits)
    scores = beam_scores[:, None] + log_probs
    scores_2d = scores.reshape(batch_size, num_beams * vocab_size)
    top_scores, top_indices = _top_k(scores_2d, 2 * num_beams)

    beam_scores = _advance_beams(
        batch_size, num_beams, vocab_size, top_scores, top_indices,
        beam_tokens, beam_scores, done, finished, eos_token_id,
        length_penalty, past_length, cache,
    )
    past_length += 1

    # Steps 2..max_length-1
    for step in range(2, max_length):
        if early_stopping and all(all(d) for d in done):
            break

        # Feed only the last generated token
        new_tokens = mx.array(
            [[beam_tokens[i][-1]] if beam_tokens[i] else [pad_token_id]
             for i in range(total_beams)],
            dtype=mx.int32,
        )

        logits = model.decode_step_cached(
            new_tokens, enc_hidden, cache,
            past_length=past_length - 1,
            encoder_attention_mask=enc_mask,
        )
        mx.eval(logits)
        _eval_cache(cache)

        next_token_logits = logits[:, -1, :]
        log_probs = _log_softmax(next_token_logits)

        scores = beam_scores[:, None] + log_probs
        scores_2d = scores.reshape(batch_size, num_beams * vocab_size)
        top_scores, top_indices = _top_k(scores_2d, 2 * num_beams)

        beam_scores = _advance_beams(
            batch_size, num_beams, vocab_size, top_scores, top_indices,
            beam_tokens, beam_scores, done, finished, eos_token_id,
            length_penalty, past_length, cache,
        )
        past_length += 1

    # Select best finished hypothesis per batch item
    results = []
    for b in range(batch_size):
        if finished[b]:
            best = max(finished[b], key=lambda x: x[0])
            results.append(best[1])
        else:
            ids = beam_tokens[b * num_beams]
            results.append([t for t in ids if t != eos_token_id and t != pad_token_id])

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eval_cache(cache: list[DecoderLayerCache]) -> None:
    """Force-evaluate all cache tensors so they're materialized for the next step."""
    for lc in cache:
        if lc.self_k is not None:
            mx.eval(lc.self_k, lc.self_v)
        if lc.cross_k is not None:
            mx.eval(lc.cross_k, lc.cross_v)


def _log_softmax(x: mx.array) -> mx.array:
    max_x = mx.max(x, axis=-1, keepdims=True)
    shifted = x - max_x
    return shifted - mx.log(mx.sum(mx.exp(shifted), axis=-1, keepdims=True))


def _top_k(x: mx.array, k: int) -> tuple[mx.array, mx.array]:
    """Return top-k values and indices along last axis."""
    sorted_indices = mx.argsort(-x, axis=-1)
    top_indices = sorted_indices[:, :k]
    batch_size = x.shape[0]
    top_values = mx.array(
        [[float(x[b, int(top_indices[b, i])]) for i in range(k)] for b in range(batch_size)]
    )
    return top_values, top_indices


def _set_index(arr: mx.array, idx: int, value: float) -> mx.array:
    """Return a new array with arr[idx] = value."""
    lst = arr.tolist()
    lst[idx] = value
    return mx.array(lst, dtype=mx.float32)
