import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

T = 3
n_embd = 4
head_size = 2

x = torch.randn(T, n_embd)

query = nn.Linear(n_embd, head_size, bias=False)
key = nn.Linear(n_embd, head_size, bias=False)
value = nn.Linear(n_embd, head_size, bias=False)

q = query(x)
k = key(x)
v = value(x)

scores = q @ k.T
mask = torch.tril(torch.ones(T, T))
masked_scores = scores.masked_fill(mask == 0, float("-inf"))
weights = F.softmax(masked_scores, dim=-1)
out = weights @ v

print("x shape:", x.shape)
print("q shape:", q.shape)
print("k shape:", k.shape)
print("v shape:", v.shape)
print()
print("scores shape:", scores.shape)
print("weights shape:", weights.shape)
print("out shape:", out.shape)
print()
print("weights:")
print(weights)
print()
print("out:")
print(out)
