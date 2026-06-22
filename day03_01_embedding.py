import torch
import torch.nn as nn

torch.manual_seed(1337)

x = torch.tensor([
    [1, 0, 2],
    [2, 3, 1],
])

vocab_size = 4
n_embd = 5
embedding = nn.Embedding(vocab_size, n_embd)

out = embedding(x)

print("x:")
print(x)
print("x shape:", x.shape)
print()
print("embedding weight shape:", embedding.weight.shape)
print("out shape:", out.shape)
print()
print("token id at x[0, 0]:", x[0, 0])
print("embedding vector at out[0, 0]:")
print(out[0, 0])
