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

T = q.shape[0]

scores = q @ k.T
mask = torch.tril(torch.ones(T, T))
masked_scores = scores.masked_fill(mask == 0, float("-inf"))
weights = F.softmax(masked_scores, dim=-1)

print("scores:")
print(scores)
print()
print("mask:")
print(mask)
print()
print("masked_scores:")
print(masked_scores)
print()
print("weights:")
print(weights)
print()
print("row sums:", weights.sum(dim=-1))
