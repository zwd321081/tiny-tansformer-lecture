# Checks

This file records understanding checks. Each check should test one small concept.

## Check 01: `set`, `list`, and `sorted`

Code:

```python
text = "banana"
chars = sorted(list(set(text)))
print(chars)
```

Prediction:

```text
["a", "b", "n"]
```

Expected result:

```text
['a', 'b', 'n']
```

Status: passed.

Reason:

`set(text)` removes duplicate characters. `sorted(...)` sorts the remaining characters.

## Check 02: repeated character removal

Code:

```python
text = "hello"
chars = sorted(list(set(text)))
print(chars)
```

Prediction:

```text
['e', 'h', 'l', 'o']
```

Expected result:

```text
['e', 'h', 'l', 'o']
```

Status: passed.

Reason:

The second `l` is removed by `set(text)`, then the unique characters are sorted.

## Check 03: `stoi` and `itos`

Code A:

```python
chars = ['e', 'h', 'l', 'o']
stoi = {ch: i for i, ch in enumerate(chars)}
print(stoi)
```

Initial prediction:

```text
{0:'e',1:'h',2:'l',3:'o'}
```

Expected result:

```text
{'e': 0, 'h': 1, 'l': 2, 'o': 3}
```

Status: corrected.

Reason:

`stoi` means string to integer, so the character is the key and the integer is the value.

Code B:

```python
chars = ['e', 'h', 'l', 'o']
itos = {i: ch for i, ch in enumerate(chars)}
print(itos)
```

Prediction:

```text
{0:'e',1:'h',2:'l',3:'o'}
```

Expected result:

```text
{0: 'e', 1: 'h', 2: 'l', 3: 'o'}
```

Status: passed.

Reason:

`itos` means integer to string, so the integer is the key and the character is the value.

## Check 04: `encode`

Setup:

```python
chars = ['e', 'h', 'l', 'o']
stoi = {ch: i for i, ch in enumerate(chars)}

def encode(s):
    ids = []
    for ch in s:
        ids.append(stoi[ch])
    return ids
```

Code A:

```python
print(encode("hello"))
```

Initial prediction:

```text
[0, 1, 2, 3]
```

Expected result:

```text
[1, 0, 2, 2, 3]
```

Status: corrected.

Reason:

`encode` preserves the input order and repeated characters: `h -> 1`, `e -> 0`, `l -> 2`, `l -> 2`, `o -> 3`.

Code B:

```python
print(encode("hole"))
```

Prediction:

```text
[1, 3, 2, 0]
```

Expected result:

```text
[1, 3, 2, 0]
```

Status: passed.

## Check 05: `decode`

Setup:

```python
chars = ['e', 'h', 'l', 'o']
itos = {i: ch for i, ch in enumerate(chars)}

def decode(ids):
    chars_out = []
    for i in ids:
        chars_out.append(itos[i])
    return "".join(chars_out)
```

Code A:

```python
print(decode([1, 0, 2, 2, 3]))
```

Initial prediction:

```text
['h','e','l','l','o']
```

Expected result:

```text
hello
```

Status: corrected.

Reason:

The intermediate `chars_out` is a list, but `decode` returns `"".join(chars_out)`, which is a string.

Code B:

```python
print(decode([1, 3, 2, 0]))
```

Prediction:

```text
hole
```

Expected result:

```text
hole
```

Status: passed.

## Check 06: changing `text` versus changing `encode(...)`

Task:

Change:

```python
text = "hello world"
```

to:

```python
text = "hello transformer"
```

Observed output:

```text
chars: [' ', 'a', 'e', 'f', 'h', 'l', 'm', 'n', 'o', 'r', 's', 't']
encoded: [4, 2, 5, 5, 8]
decoded: hello
```

Question:

Why does `encoded` / `decoded` still correspond to `"hello"`?

Answer:

Changing `text` changes the character table and mappings. It does not change the later call to `encode("hello")`.

Status: passed.

## Check 07: encoding and decoding `transformer`

Setup:

```python
text = "hello transformer"
encoded = encode("transformer")
decoded = decode(encoded)
```

Observed output:

```text
encoded: [11, 9, 1, 7, 10, 3, 8, 9, 6, 2, 9]
decoded: transformer
```

Questions:

1. Why does `r` appear three times as `9`?
2. Why can `decoded` become `transformer` again?

Answer:

1. `transformer` contains three `r` characters, and `stoi['r']` is `9`.
2. `decode` maps integers back to characters. The exact mapping is decided by `text`, which defines the vocabulary and `itos`.

Status: passed.

## Check 08: next-character training target

Code:

```python
text = "transformer"
x = text[:-1]
y = text[1:]
```

Prediction:

```text
x = transforme
y = ransformer
```

Expected result:

```text
x = transforme
y = ransformer
```

Status: passed.

Reason:

`x` removes the last character. `y` removes the first character, so each position in `x` lines up with the next character in `y`.

## Check 09: next-character pairs for `hello`

Task:

Change:

```python
text = "transformer"
```

to:

```python
text = "hello"
```

Observed output:

```text
text: hello
x: hell
y: ello

training pairs:
h -> e
e -> l
l -> l
l -> o
```

Status: passed.

Reason:

`x` and `y` are offset by one character. Each character in `x` is paired with the next character from `y`.

## Check 10: numeric next-character pairs

Setup:

```python
text = "hello"
chars = ['e', 'h', 'l', 'o']
stoi = {'e': 0, 'h': 1, 'l': 2, 'o': 3}
x = "hell"
y = "ello"
```

Code:

```python
x_ids = [stoi[ch] for ch in x]
y_ids = [stoi[ch] for ch in y]
```

Prediction:

```text
x_ids = [1, 0, 2, 2]
y_ids = [0, 2, 2, 3]
```

Expected result:

```text
x_ids = [1, 0, 2, 2]
y_ids = [0, 2, 2, 3]
```

Status: passed.

Reason:

Each character is replaced by its integer from `stoi`, while the original order is preserved.

## Check 11: 2D tensor shape and indexing

Code:

```python
batch = torch.tensor([
    [1, 0, 2, 2],
    [0, 2, 2, 3],
])
```

Prediction:

```text
batch.shape = (2, 4)
batch[0] = [1, 0, 2, 2]
batch[1] = [0, 2, 2, 3]
batch[0, 2] = 2
```

Status: passed.

Reason:

The tensor has 2 rows and 4 columns. `batch[0, 2]` means row 0, column/index 2.

## Check 12: dimensions versus shape values

Code:

```python
a = [10, 20, 30]
b = [
    [10, 20, 30],
]
c = [
    [10, 20, 30],
    [40, 50, 60],
]
```

Prediction:

```text
a: 1D, shape [3]
b: 2D, shape [1, 3]
c: 2D, shape [2, 3]
```

Status: passed.

Reason:

The number of dimensions is the number of numbers in `shape`. The value of each shape number is the length along that dimension.

## Check 13: PyTorch tensor shape and indexing

Code:

```python
example = torch.tensor([
    [10, 20, 30],
    [40, 50, 60],
])

print("example shape:", example.shape)
print("example[1, 2]:", example[1, 2])
```

Prediction:

```text
example.shape = [2, 3]
example[1, 2] = 60
```

Observed output:

```text
example shape: torch.Size([2, 3])
example[1, 2]: tensor(60)
```

Status: passed.

Reason:

There are 2 rows and 3 columns. Row 1, column 2 is `60`.

## Check 14: `[B, T]` language model shape

Code:

```python
x = torch.tensor([
    [4, 2, 5, 5],
    [11, 9, 1, 7],
    [10, 3, 8, 9],
])
```

Prediction:

```text
x.shape = [3, 4]
B = 3
T = 4
sample 0 = [4, 2, 5, 5]
sample 2 = [10, 3, 8, 9]
x[2, 1] = 3
```

Status: passed.

Reason:

For language model inputs, a 2D tensor is often read as `[B, T]`: `B` samples, each with `T` token ids.

## Check 15: slicing one training sample from token data

Code:

```python
data = torch.tensor([1, 0, 2, 2, 3, 1, 2, 0])
block_size = 4

x = data[2:2+block_size]
y = data[3:3+block_size]
```

Prediction:

```text
x = [2, 2, 3, 1]
y = [2, 3, 1, 2]
x.shape = [4]
y.shape = [4]
```

Status: passed.

Reason:

`x` starts at index 2 and has length 4. `y` starts one position later and also has length 4.

## Check 16: stacking samples into `[B, T]`

Code:

```python
data = torch.tensor([1, 0, 2, 2, 3, 1, 2, 0])
block_size = 3

x1 = data[0:0+block_size]
x2 = data[3:3+block_size]
x = torch.stack([x1, x2])
```

Prediction:

```text
x1 = [1, 0, 2]
x2 = [2, 3, 1]
x = [[1, 0, 2], [2, 3, 1]]
x.shape = [2, 3]
B = 2
T = 3
```

Status: passed.

Reason:

Each sample has length `T = 3`. Stacking 2 samples creates a 2D tensor with `B = 2`, so shape is `[2, 3]`.

## Check 17: embedding output shape

Code A:

```python
x = torch.tensor([
    [1, 0, 2],
    [2, 3, 1],
])
vocab_size = 4
n_embd = 5
embedding = nn.Embedding(vocab_size, n_embd)
out = embedding(x)
```

Initial prediction:

```text
x.shape = [2, 3]
embedding.weight.shape = unknown
out.shape = [4, 5]
out[0, 0] = first value of the weight matrix
```

Expected result:

```text
x.shape = [2, 3]
embedding.weight.shape = [4, 5]
out.shape = [2, 3, 5]
out[0, 0] is a 5-number vector
```

Status: corrected.

Reason:

`embedding.weight` is the lookup table. `out = embedding(x)` keeps the `[B, T]` positions and replaces each token id with a length-`C` vector.

Code B:

```python
x = torch.tensor([
    [2, 0],
])
vocab_size = 3
n_embd = 4
embedding = nn.Embedding(vocab_size, n_embd)
out = embedding(x)
```

Prediction:

```text
x.shape = [1, 2]
embedding.weight.shape = [3, 4]
out.shape = [1, 2, 4]
out[0, 0] == embedding.weight[2]
```

Status: passed.

Reason:

`x[0, 0]` is token id `2`, so `out[0, 0]` is the vector from row `2` of the embedding table.

## Check 18: embedding coordinates versus token ids

Code:

```python
x = torch.tensor([
    [3, 5, 3],
    [0, 9, 1],
])
embedding = nn.Embedding(10, 6)
out = embedding(x)
```

Prediction:

```text
x.shape = [2, 3]
out.shape = [2, 3, 6]
out[0, 0] uses embedding.weight row 3
out[0, 2] uses embedding.weight row 3
```

Status: corrected.

Reason:

`[0, 0]` is a coordinate. `x[0, 0]` is the token id at that coordinate. `out[0, 0]` is the embedding vector looked up from that token id.

Core formula:

```python
out[b, t] == embedding.weight[x[b, t]]
```

## Check 19: verifying embedding lookup at a different coordinate

Setup:

```python
x = torch.tensor([
    [2, 1],
    [0, 2],
])
b = 1
t = 1
```

Prediction:

```text
x[1, 1] = tensor(2)
out[1, 1] = embedding.weight[2]
same = True
```

Observed result:

```text
x[b, t]: tensor(2)
same: True
```

Status: passed.

Reason:

Coordinate `[1, 1]` contains token id `2`, so the embedding output at `[1, 1]` is row `2` of the embedding table.

## Check 20: position embedding shapes

Code:

```python
x = torch.tensor([
    [2, 1, 0],
    [0, 2, 1],
])
B, T = x.shape
n_embd = 4
positions = torch.arange(T)
```

Prediction:

```text
B = 2
T = 3
positions = [0, 1, 2]
token_vectors.shape = [2, 3, 4]
position_vectors.shape = [3, 4]
combined.shape = [2, 3, 4]
combined[1, 2] = token_vectors[1, 2] + position_vectors[2]
```

Status: corrected.

Reason:

`T` is the sequence length. `positions` is a 1D tensor of position ids `[0, 1, 2]`; after position embedding it becomes `[T, C]`.

## Check 21: same token at different positions

Code:

```python
x = torch.tensor([
    [2, 1, 2],
])

combined[0, 0] = token_vectors[0, 0] + position_vectors[0]
combined[0, 2] = token_vectors[0, 2] + position_vectors[2]
```

Prediction:

```text
combined[0, 0] and combined[0, 2] are different.
```

Status: passed.

Reason:

The token vectors are the same because both token ids are `2`, but the position vectors are different: position `0` versus position `2`.

## Check 22: same token at the same position across batches

Code:

```python
x = torch.tensor([
    [2, 1, 0],
    [0, 1, 2],
])
```

Question:

```text
Are combined[0, 1] and combined[1, 1] the same?
```

Answer:

```text
Yes.
```

Status: corrected.

Reason:

Both positions contain token id `1`, so both token vectors come from `token_embedding.weight[1]`. Both are at position `t = 1`, so both add `position_vectors[1]`. The batch index `b` does not change the position vector.

## Check 23: linear logits shape

Setup:

```python
B = 4
T = 8
C = 16
vocab_size = 10
combined.shape = [B, T, C]
lm_head = nn.Linear(C, vocab_size)
logits = lm_head(combined)
```

Prediction:

```text
combined.shape = [4, 8, 16]
logits.shape = [4, 8, 10]
logits[2, 5].shape = [10]
```

Status: corrected.

Reason:

`logits[2, 5]` is not a token id. It is the vector of scores for all `vocab_size` possible next tokens at batch row `2`, time position `5`.

## Check 24: interpreting logits scores

Code:

```python
logits[0, 1] = torch.tensor([0.2, -1.0, 3.4, 0.5, 1.1])
```

Prediction:

```text
There are 5 scores.
The highest score is 3.4 at index 2.
The model most prefers token id 2.
```

Status: passed.

Reason:

Each index in the logits vector corresponds to one candidate token id. The highest score is the model's current strongest prediction.

## Check 25: softmax probabilities

Code:

```python
logits = torch.tensor([1.0, 4.0, 0.5])
probs = softmax(logits)
```

Prediction:

```text
probs has 3 numbers.
probs sums to 1.
token id 1 has the highest probability.
```

Status: passed.

Reason:

The largest logit is `4.0` at index `1`, so token id `1` gets the highest softmax probability.

## Check 26: cross entropy target token

Code:

```python
logits = torch.tensor([[1.0, 4.0, 0.5]])
target = torch.tensor([2])
```

Answer:

```text
The correct token id is 2.
The model most prefers token id 1.
The loss is large.
```

Status: corrected.

Reason:

The candidate token ids are `0`, `1`, and `2`, but `target[0] = 2` selects token id `2` as the correct answer. Cross entropy is large because the model gives the correct token a low score and therefore a low probability.
