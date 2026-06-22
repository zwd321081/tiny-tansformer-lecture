import torch
import torch.nn.functional as F

T = 4

scores = torch.zeros(T, T)
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
print("weights after softmax:")
print(weights)
print()
print("row sums:", weights.sum(dim=-1))
