import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

B = 2
T = 3
C = 4
head_size = 5

x = torch.randn(B, T, C)

query = nn.Linear(C, head_size, bias=False)
key = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

q = query(x)
k = key(x)
v = value(x)

scores = q @ k.transpose(-2, -1)
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
print("mask shape:", mask.shape)
print("masked_scores shape:", masked_scores.shape)
print("weights shape:", weights.shape)
print("out shape:", out.shape)
