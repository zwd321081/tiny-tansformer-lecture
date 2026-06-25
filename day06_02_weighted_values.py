import torch

weights = torch.tensor(
    [
        [1.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.2, 0.3, 0.5],
    ]
)

scalar_values = torch.tensor([4.0, 10.0, 20.0])

scalar_out = weights @ scalar_values

print("weights shape:", weights.shape)
print("scalar_values shape:", scalar_values.shape)
print("scalar_out:", scalar_out)
print()

vector_values = torch.tensor(
    [
        [4.0, 1.0],
        [10.0, 2.0],
        [20.0, 3.0],
    ]
)

vector_out = weights @ vector_values

print("vector_values shape:", vector_values.shape)
print("vector_out shape:", vector_out.shape)
print("vector_out:")
print(vector_out)
print()

causal_weights = torch.tensor(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
        [1 / 3, 1 / 3, 1 / 3, 0.0],
        [0.25, 0.25, 0.25, 0.25],
    ]
)

causal_values = torch.tensor([10.0, 20.0, 30.0, 40.0])
causal_out = causal_weights @ causal_values

print("causal_weights shape:", causal_weights.shape)
print("causal_values shape:", causal_values.shape)
print("causal_out:", causal_out)
