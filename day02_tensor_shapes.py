import torch

ids = [1, 0, 2, 2]
x = torch.tensor(ids)

print("ids:", ids)
print("x:", x)
print("x shape:", x.shape)
print("x first item:", x[0])
print()

batch = torch.tensor(
    [
        [1, 0, 2, 2],
        [0, 2, 2, 3],
    ]
)

print("batch:")
print(batch)
print("batch shape:", batch.shape)
print("batch first row:", batch[0])
print("batch second row:", batch[1])
print("batch row 0 col 2:", batch[0, 2])


example = torch.tensor(
    [
        [10, 20, 30],
        [40, 50, 60],
    ]
)

print()
print("example:")
print(example)
print("example shape:", example.shape)
print("example[1, 2]:", example[1, 2])
