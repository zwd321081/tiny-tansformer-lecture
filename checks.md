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
