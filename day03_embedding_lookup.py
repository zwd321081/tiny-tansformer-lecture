import torch
import torch.nn as nn

torch.manual_seed(1337)

x = torch.tensor([
    [2, 1],
    [0, 2],
])

embedding = nn.Embedding(3, 4)
out = embedding(x)

b = 0
t = 0
token_id = x[b, t]

print("x:")
print(x)
print("x shape:", x.shape)
print()
print("embedding.weight:")
print(embedding.weight)
print("embedding.weight shape:", embedding.weight.shape)
print()
print("out shape:", out.shape)
print()
print("b:", b)
print("t:", t)
print("x[b, t]:", token_id)
print("out[b, t]:")
print(out[b, t])
print("embedding.weight[x[b, t]]:")
print(embedding.weight[token_id])
print("same:", torch.equal(out[b, t], embedding.weight[token_id]))
