import torch

x = torch.tensor(
    [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ]
)

attention_out = torch.tensor(
    [
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
    ]
)

out = x + attention_out

print("x shape:", x.shape)
print("attention_out shape:", attention_out.shape)
print("out shape:", out.shape)
print()
print("out:")
print(out)
