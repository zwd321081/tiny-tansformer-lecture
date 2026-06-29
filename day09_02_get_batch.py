import torch

torch.manual_seed(1337)

text = "hello transformer"

chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[ch] for ch in s]


data = torch.tensor(encode(text), dtype=torch.long)

batch_size = 4
block_size = 3


def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y, ix


x, y, ix = get_batch()

print("data:", data)
print("data length:", len(data))
print("batch_size:", batch_size)
print("block_size:", block_size)
print("start indices:", ix)
print()
print("x:")
print(x)
print("y:")
print(y)
print("x shape:", x.shape)
print("y shape:", y.shape)
