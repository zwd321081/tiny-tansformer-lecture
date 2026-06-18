import torch

data = torch.tensor([1, 0, 2, 2, 3, 1, 2, 0])
block_size = 3

x1 = data[0:0 + block_size]
x2 = data[3:3 + block_size]
x = torch.stack([x1, x2])

y1 = data[1:1 + block_size]
y2 = data[4:4 + block_size]
y = torch.stack([y1, y2])

print("data:", data)
print("x1:", x1)
print("x2:", x2)
print("x:")
print(x)
print("x shape:", x.shape)
print()
print("y1:", y1)
print("y2:", y2)
print("y:")
print(y)
print("y shape:", y.shape)
