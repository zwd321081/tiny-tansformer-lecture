import torch
import torch.nn as nn

x = torch.tensor(
    [
        [1.0, 2.0, 3.0],
        [10.0, 20.0, 30.0],
    ]
)

layer_norm = nn.LayerNorm(3)
out = layer_norm(x)

print("x shape:", x.shape)
print("out shape:", out.shape)
print()
print("x:")
print(x)
print()
print("out:")
print(out)
print()
print("out row means:", out.mean(dim=-1))
print("out row stds:", out.std(dim=-1, unbiased=False))
print()

batch_x = torch.randn(2, 3, 8)
batch_layer_norm = nn.LayerNorm(8)
batch_out = batch_layer_norm(batch_x)

print("batch_x shape:", batch_x.shape)
print("batch_out shape:", batch_out.shape)
print("batch_out means over n_embd:", batch_out.mean(dim=-1))
print("batch_out stds over n_embd:", batch_out.std(dim=-1, unbiased=False))
