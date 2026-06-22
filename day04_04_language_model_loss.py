import torch
import torch.nn.functional as F

logits = torch.tensor([
    [
        [1.0, 4.0, 0.5],
        [2.0, 0.0, 1.0],
    ],
    [
        [0.5, 0.1, 3.0],
        [0.2, 2.5, 0.3],
    ],
])

targets = torch.tensor([
    [1, 0],
    [2, 2],
])

B, T, V = logits.shape

flat_logits = logits.view(B * T, V)
flat_targets = targets.view(B * T)

loss = F.cross_entropy(flat_logits, flat_targets)

print("logits shape:", logits.shape)
print("targets shape:", targets.shape)
print()
print("flat_logits shape:", flat_logits.shape)
print("flat_targets shape:", flat_targets.shape)
print()
print("flat_logits:")
print(flat_logits)
print("flat_targets:", flat_targets)
print()
print("loss:", loss)
