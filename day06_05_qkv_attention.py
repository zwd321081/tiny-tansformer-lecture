import torch
import torch.nn.functional as F

q = torch.tensor(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ]
)

k = torch.tensor(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ]
)

v = torch.tensor(
    [
        [4.0, 1.0],
        [10.0, 2.0],
        [20.0, 3.0],
    ]
)

T = q.shape[0]

scores = q @ k.T
mask = torch.tril(torch.ones(T, T))
masked_scores = scores.masked_fill(mask == 0, float("-inf"))
weights = F.softmax(masked_scores, dim=-1)
out = weights @ v

print("q shape:", q.shape)
print("k shape:", k.shape)
print("v shape:", v.shape)
print()
print("weights shape:", weights.shape)
print("out shape:", out.shape)
print()
print("weights:")
print(weights)
print()
print("out:")
print(out)
