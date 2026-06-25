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
n_embd = 8
num_heads = 4
head_size = 2
block_size = 8

x = torch.randn(B, T, n_embd)
heads = nn.ModuleList(
    [Head(n_embd, head_size, block_size) for _ in range(num_heads)]
)
proj = nn.Linear(num_heads * head_size, n_embd)

head_outputs = [head(x) for head in heads]
concat_out = torch.cat(head_outputs, dim=-1)
projected_out = proj(concat_out)

print("x shape:", x.shape)
print("single head output shape:", head_outputs[0].shape)
print("num heads:", len(head_outputs))
print("concat out shape:", concat_out.shape)
print("projected out shape:", projected_out.shape)
