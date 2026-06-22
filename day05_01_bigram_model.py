import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

x = torch.tensor([
    [1, 0, 2],
    [2, 3, 1],
])

targets = torch.tensor([
    [0, 2, 2],
    [3, 1, 2],
])

B, T = x.shape
vocab_size = 4

token_embedding_table = nn.Embedding(vocab_size, vocab_size)
logits = token_embedding_table(x)

flat_logits = logits.view(B * T, vocab_size)
flat_targets = targets.view(B * T)
loss = F.cross_entropy(flat_logits, flat_targets)

print("x shape:", x.shape)
print("targets shape:", targets.shape)
print("logits shape:", logits.shape)
print()
print("flat_logits shape:", flat_logits.shape)
print("flat_targets shape:", flat_targets.shape)
print("loss:", loss)
