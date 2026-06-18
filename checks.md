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
