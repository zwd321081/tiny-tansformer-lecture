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
