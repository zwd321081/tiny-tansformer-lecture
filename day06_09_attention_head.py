import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)


class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        scores = q @ k.transpose(-2, -1)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = weights @ v

        return out


B = 2
T = 3
n_embd = 4
head_size = 2
block_size = 8

x = torch.randn(B, T, n_embd)
head = Head(n_embd, head_size, block_size)
out = head(x)

print("x shape:", x.shape)
print("out shape:", out.shape)
print("tril shape:", head.tril.shape)
print("used mask shape:", head.tril[:T, :T].shape)
