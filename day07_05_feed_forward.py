import torch
import torch.nn as nn

torch.manual_seed(1337)

B = 2
T = 3
n_embd = 8
hidden_size = 4 * n_embd

x = torch.randn(B, T, n_embd)

feed_forward = nn.Sequential(
    nn.Linear(n_embd, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, n_embd),
)

out = feed_forward(x)

print("x shape:", x.shape)
print("hidden_size:", hidden_size)
print("out shape:", out.shape)
