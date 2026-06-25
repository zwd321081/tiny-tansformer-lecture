import torch

q = torch.tensor(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ]
)

k = torch.tensor(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ]
)

scores = q @ k.T

print("q shape:", q.shape)
print("k shape:", k.shape)
print("k.T shape:", k.T.shape)
print("scores shape:", scores.shape)
print("scores:")
print(scores)
