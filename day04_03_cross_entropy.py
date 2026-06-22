import torch
import torch.nn.functional as F

logits = torch.tensor([
    [1.0, 4.0, 0.5],
])
target = torch.tensor([1])

loss = F.cross_entropy(logits, target)
probs = F.softmax(logits, dim=-1)

print("logits:")
print(logits)
print("logits shape:", logits.shape)
print("target:", target)
print("target shape:", target.shape)
print("loss:", loss)
print()
print("probs:")
print(probs)
print("probability assigned to correct token:", probs[0, target[0]])
