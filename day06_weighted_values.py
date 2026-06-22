import torch

weights = torch.tensor([
    [1.0, 0.0, 0.0],
    [0.5, 0.5, 0.0],
    [0.2, 0.3, 0.5],
])

values = torch.tensor([4.0, 10.0, 20.0])
out = weights @ values

print("weights:")
print(weights)
print()
print("values:", values)
print("out:", out)
