import torch

text = "hello transformer"

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[ch] for ch in s]


def decode(ids):
    return "".join(itos[i] for i in ids)


data = torch.tensor(encode(text), dtype=torch.long)
x = data[:-1]
y = data[1:]

print("text:", text)
print("chars:", chars)
print("vocab_size:", vocab_size)
print("encoded data:", data)
print("decoded data:", decode(data.tolist()))
print()
print("x:", x)
print("y:", y)
print("x shape:", x.shape)
print("y shape:", y.shape)
