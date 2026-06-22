# Progress

## Current Goal

Build Python and PyTorch foundations for learning Transformer from the top down.

## Current Stage

Stage 6: causal mask and attention prep.

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
- Passed Check 33: understood causal visibility for self-attention.
- Corrected and passed Check 34: understood causal mask is a full `T x T` matrix.
- Corrected Check 35: understood `masked_scores` keeps score shape and original visible scores, replacing only masked positions with `-inf`.
- Corrected Check 36: understood masked softmax weights and that softmax amplifies larger scores.
- Corrected Check 37: understood `weights @ values` performs weighted sums row by row.
- Corrected and passed Check 38: understood Q/K/V attention shapes.
- Passed Check 39: understood the minimal Q/K/V attention chain shapes.
- Corrected Check 40: understood batched Q/K/V shapes, value dimension in `out`, and mask shape `[T, T]`.
- Corrected Check 41: understood scores are divided by `sqrt(head_size)`.
- Passed Check 42: understood single `Head` shape flow.
- Passed Check 43: understood multi-head attention concatenates head outputs on the last dimension.

## Current Task

Run and understand the first Python exercises:

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
uv run python day06_01_causal_mask.py
uv run python day06_02_weighted_values.py
uv run python day06_03_qkv_attention.py
uv run python day06_04_batched_attention.py
uv run python day06_05_attention_head.py
uv run python day06_06_multi_head_attention.py
```

## Next Step

Understand these Python concepts from the first exercise:

- string
- `set`
- `list`
- `sorted`
- `dict`
- function
- `for` loop

Then implement `encode` and `decode` with small changes by hand.

Also continue using `checks.md` to verify understanding with prediction, code changes, and reverse explanation.

After that, connect `day01_02_next_char.py` to numeric encoding, so training pairs become integer pairs.

Then understand:

- how a Python list becomes a `torch.tensor`
- what `shape` means
- how 1D and 2D tensors are indexed
- how `nn.Embedding` turns `[B, T]` token ids into `[B, T, C]` vectors
- how `out[b, t] == embedding.weight[x[b, t]]`
- how token embeddings and position embeddings are added
- how `nn.Linear` turns `[B, T, C]` into `[B, T, vocab_size]` logits
- how softmax turns logits into probabilities
- how cross entropy compares logits against the target token id
- how language model loss flattens `[B, T, V]` logits and `[B, T]` targets
- how a Bigram model maps token ids directly to next-token logits
- how `zero_grad`, `backward`, and `optimizer.step` reduce loss
- how a language model generates one token at a time
- how causal masks prevent positions from attending to future tokens
- how attention weights aggregate values with `weights @ values`
- how Q/K/V form minimal causal self-attention
- how batched Q/K/V attention works on `[B, T, C]`
- how to wrap single-head causal self-attention in a `Head` module
- how multi-head attention concatenates several `Head` outputs

## Notes For Next Session

Continue slowly. The learner is new to Python and PyTorch, so explain in small chunks and avoid giving full Transformer code.
