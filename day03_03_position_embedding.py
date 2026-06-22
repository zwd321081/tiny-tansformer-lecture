import torch
import torch.nn as nn

torch.manual_seed(1337)

x = torch.tensor([
    [2, 1, 0],
    [0, 2, 1],
])

B, T = x.shape
vocab_size = 3
n_embd = 4

token_embedding = nn.Embedding(vocab_size, n_embd)
position_embedding = nn.Embedding(T, n_embd)

token_vectors = token_embedding(x)
positions = torch.arange(T)
position_vectors = position_embedding(positions)

combined = token_vectors + position_vectors

print("x:")
print(x)
print("x shape:", x.shape)
print()
print("positions:", positions)
print("positions shape:", positions.shape)
print()
print("token_vectors shape:", token_vectors.shape)
print("position_vectors shape:", position_vectors.shape)
print("combined shape:", combined.shape)
print()
print("token_vectors[0, 0]:")
print(token_vectors[0, 0])
print("position_vectors[0]:")
print(position_vectors[0])
print("combined[0, 0]:")
print(combined[0, 0])
print("same as sum:", torch.equal(combined[0, 0], token_vectors[0, 0] + position_vectors[0]))
