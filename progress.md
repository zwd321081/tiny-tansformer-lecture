# Progress

## Current Goal

Build Python and PyTorch foundations for learning Transformer from the top down.

## Current Stage

Stage 6: restart Transformer from attention fundamentals.

## Completed

- Created persistent learning project directory.
- Chose the learning rule: write core code by hand, use AI for explanation, hints, checking, and debugging.
- Passed Check 01: predicted `sorted(list(set("banana")))`.
- Passed Check 02: predicted `sorted(list(set("hello")))`.
- Corrected Check 03: learned the direction difference between `stoi` and `itos`.
- Corrected and passed Check 04: `encode` preserves input order and repeated characters.
- Corrected and passed Check 05: `decode` maps integers back to characters and joins them into a string.
- Passed Check 06: understood that `text` controls the vocabulary, while `encode("hello")` controls what gets encoded.
- Passed Check 07: explained repeated encoded values and how `decode` reconstructs `transformer`.
- Passed Check 08: understood `x = text[:-1]` and `y = text[1:]` as next-character training pairs.
- Passed Check 09: produced next-character training pairs for `hello`.
- Passed Check 10: converted character training pairs into numeric training pairs.
- Passed Check 11: understood 2D tensor shape and indexing.
- Passed Check 12: separated dimension count from shape values.
- Passed Check 13: verified tensor shape and indexing in PyTorch.
- Passed Check 14: interpreted a 2D language model tensor as `[B, T]`.
- Passed Check 15: sliced one `[T]` training sample and its next-token target from token data.
- Passed Check 16: stacked multiple `[T]` samples into a `[B, T]` batch.
- Corrected and passed Check 17: understood `nn.Embedding` table shape and output shape `[B, T, C]`.
- Corrected Check 18: separated embedding coordinates from token ids with `out[b, t] == embedding.weight[x[b, t]]`.
- Passed Check 19: verified embedding lookup at coordinate `[1, 1]`.
- Corrected Check 20: understood position ids, position vector shape, and `combined[b, t] = token_vectors[b, t] + position_vectors[t]`.
- Passed Check 21: understood why the same token gets different combined vectors at different positions.
- Corrected Check 22: understood same token and same position across different batch rows gives the same combined vector.
- Corrected Check 23: understood `logits[b, t]` is a score vector over the vocabulary, not a token id.
- Passed Check 24: interpreted the highest logit as the most preferred token id.
- Passed Check 25: understood softmax probabilities sum to 1 and preserve the highest-logit token as highest probability.
- Corrected Check 26: understood `target` stores the correct token id and cross entropy depends on the probability assigned to that correct token.
- Passed Check 27: flattened `[B, T, V]` logits into `[B*T, V]` and `[B, T]` targets into `[B*T]` for loss.
- Corrected Check 28: separated `B*T` prediction tasks from `V` candidate token scores.
- Corrected Check 29: separated Bigram embedding table shape `[V, V]` from logits shape `[B, T, V]`.
- Passed Check 30: observed Bigram loss decreasing and weights changing after training.
- Corrected Check 31: understood generation uses `logits[:, -1, :]` with shape `[B, V]`.
- Corrected Check 32: understood `logits[b, t]` is a vocabulary score vector and `logits[b, t, token_id]` is one token's score.
- Restarted Day 6 from Transformer attention fundamentals.
- Passed Check R6-01: causal visibility and full `T = 4` mask.
- Corrected and passed Check R6-02: predicted `scores`, `mask`, `masked_scores`, and softmax `weights` for `day06_01_causal_mask.py`.
- Passed Check R6-03: predicted and verified `out = weights @ values` in `day06_02_weighted_values.py`.
- Passed Check R6-04: connected `causal_weights @ causal_values` to context mixing and understood why future values must not affect earlier outputs.
- Corrected and passed Check R6-05: understood why `scores = q @ k.T` gives a `[T, T]` query-key matching table.
- Passed Check R6-06: predicted masked Q/K scores and interpreted softmax attention weights in `day06_04_masked_attention_weights.py`.
- Passed Check R6-07: understood `out = weights @ v` as the self-attention output, a context-mixed representation used before logits.
- Corrected and passed Check R6-08: understood PyTorch `nn.Linear(n_embd, head_size)` stores weight as `[head_size, n_embd]` and produces Q/K/V shape `[T, head_size]`.
- Passed Check R6-09: assembled minimal single-head causal self-attention from `x -> q/k/v -> scores -> weights -> out`.
- Corrected and passed Check R6-10: added batch dimension and understood batched attention shapes `[B, T, head_size] @ [B, head_size, T] -> [B, T, T]`.
- Passed Check R6-11: wrapped batched causal self-attention into a reusable `Head` module with output shape `[B, T, head_size]`.
- Corrected and passed Check R6-12: understood multi-head concat shape `[B, T, num_heads * head_size]` and projection back to `[B, T, n_embd]`.
- Passed Check R6-13: wrapped multi-head causal self-attention into a reusable `MultiHeadAttention` module.

## Current Task

Restart Transformer learning from Day 6. Keep Day 1-5 as completed foundation:

```bash
python3 day01_01_python_basics.py
python3 day01_02_next_char.py
uv run python day02_01_tensor_shapes.py
uv run python day02_02_batch.py
uv run python day03_01_embedding.py
uv run python day03_02_embedding_lookup.py
uv run python day03_03_position_embedding.py
uv run python day04_01_linear_logits.py
uv run python day04_02_softmax.py
uv run python day04_03_cross_entropy.py
uv run python day04_04_language_model_loss.py
uv run python day05_01_bigram_model.py
uv run python day05_02_bigram_train.py
uv run python day05_03_bigram_generate.py
```

## Next Step

Rebuild Day 6 as the Transformer starting point:

1. Explain why Bigram is not enough and why context is needed.
2. Relearn causal visibility: each position can see itself and the past, not the future.
3. Build causal mask from scratch.
4. Created `day06_02_weighted_values.py` and understood attention as weighted averaging.
5. Created `day06_03_qk_scores.py` to show how attention scores are produced.
6. Created `day06_04_masked_attention_weights.py` to turn masked Q/K scores into attention weights.
7. Created `day06_05_qkv_attention.py` to combine weights with V values and produce attention output.
8. Created `day06_06_qkv_from_x.py` to produce Q/K/V from input `x`.
9. Created `day06_07_single_head_attention.py` as a minimal no-batch single-head causal self-attention.
10. Created `day06_08_batched_attention.py` and understood `[B, T, C]` attention shapes.
11. Created `day06_09_attention_head.py` and wrapped batched causal self-attention into a reusable `Head` module.
12. Created `day06_10_multi_head_attention.py` to concatenate several heads and project back to `n_embd`.
13. Created `day06_11_multi_head_module.py` as a reusable `MultiHeadAttention` module.
14. Next, review Day 6 attention end-to-end before moving to Transformer Block.

Do not continue to Transformer Block until Day 6 attention is rebuilt cleanly.

## Notes For Next Session

Day 6 is intentionally reset. Continue slowly from Transformer/attention fundamentals. The old `day06_02` through `day06_06` reference files were deleted so the concepts can be rebuilt in order instead of assumed.
