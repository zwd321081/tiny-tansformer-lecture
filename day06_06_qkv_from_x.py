import torch
import torch.nn as nn

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

print("x shape:", x.shape)
print("query weight shape:", query.weight.shape)
print("key weight shape:", key.weight.shape)
print("value weight shape:", value.weight.shape)
print()
print("q shape:", q.shape)
print("k shape:", k.shape)
print("v shape:", v.shape)
