import torch
import torch.nn as nn

torch.manual_seed(1337)

B = 2
T = 3
n_embd = 8
hidden_size = 4 * n_embd

x = torch.randn(B, T, n_embd)

layer_norm = nn.LayerNorm(n_embd)
feed_forward = nn.Sequential(
    nn.Linear(n_embd, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, n_embd),
)

normed_x = layer_norm(x)
ffn_out = feed_forward(normed_x)
out = x + ffn_out

print("x shape:", x.shape)
print("normed_x shape:", normed_x.shape)
print("ffn_out shape:", ffn_out.shape)
print("out shape:", out.shape)
