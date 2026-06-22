import torch
import torch.nn as nn

torch.manual_seed(1337)

x = torch.tensor([
    [2, 1, 0],
    [0, 1, 2],
])

B, T = x.shape
vocab_size = 3
n_embd = 4

token_embedding = nn.Embedding(vocab_size, n_embd)
position_embedding = nn.Embedding(T, n_embd)
lm_head = nn.Linear(n_embd, vocab_size)

token_vectors = token_embedding(x)
positions = torch.arange(T)
position_vectors = position_embedding(positions)
combined = token_vectors + position_vectors
logits = lm_head(combined)

print("x shape:", x.shape)
print("combined shape:", combined.shape)
print("logits shape:", logits.shape)
print()
print("logits[0, 0]:")
print(logits[0, 0])
print("logits[0, 0] shape:", logits[0, 0].shape)
