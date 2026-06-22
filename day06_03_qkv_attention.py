import torch
import torch.nn.functional as F

torch.manual_seed(1337)

T = 3
key_dim = 2
value_dim = 4

q = torch.randn(T, key_dim)
k = torch.randn(T, key_dim)
v = torch.randn(T, value_dim)

scores = q @ k.T
mask = torch.tril(torch.ones(T, T))
masked_scores = scores.masked_fill(mask == 0, float("-inf"))
weights = F.softmax(masked_scores, dim=-1)
out = weights @ v

print("q shape:", q.shape)
print("k shape:", k.shape)
print("v shape:", v.shape)
print()
print("scores shape:", scores.shape)
print("mask shape:", mask.shape)
print("masked_scores shape:", masked_scores.shape)
print("weights shape:", weights.shape)
print("out shape:", out.shape)
print()
print("weights:")
print(weights)
print()
print("out:")
print(out)
