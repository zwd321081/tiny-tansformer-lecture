# Progress

## Current Goal

Build Python and PyTorch foundations for learning Transformer from the top down.

## Current Stage

Stage 3: embedding basics.

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

## Current Task

Run and understand the first Python exercises:

```bash
python3 day01_python_basics.py
python3 day01_next_char.py
uv run python day02_tensor_shapes.py
uv run python day02_batch.py
uv run python day03_embedding.py
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

After that, connect `day01_next_char.py` to numeric encoding, so training pairs become integer pairs.

Then understand:

- how a Python list becomes a `torch.tensor`
- what `shape` means
- how 1D and 2D tensors are indexed
- how `nn.Embedding` turns `[B, T]` token ids into `[B, T, C]` vectors

## Notes For Next Session

Continue slowly. The learner is new to Python and PyTorch, so explain in small chunks and avoid giving full Transformer code.
