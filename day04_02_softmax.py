import torch
import torch.nn.functional as F

logits = torch.tensor([0.2, -1.0, 3.4, 0.5, 1.1])
probs = F.softmax(logits, dim=0)

print("logits:", logits)
print("probs:", probs)
print("probs sum:", probs.sum())
print("best token id:", torch.argmax(probs))
