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


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, block_size):
        super().__init__()
        head_size = n_embd // num_heads
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(num_heads * head_size, n_embd)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, num_heads, block_size):
        super().__init__()
        self.self_attention = MultiHeadAttention(n_embd, num_heads, block_size)
        self.feed_forward = FeedForward(n_embd)
        self.layer_norm_1 = nn.LayerNorm(n_embd)
        self.layer_norm_2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.self_attention(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


B = 2
T = 3
vocab_size = 4
block_size = 8
n_embd = 8
num_heads = 4

idx = torch.tensor(
    [
        [1, 0, 2],
        [2, 3, 1],
    ]
)

targets = torch.tensor(
    [
        [0, 2, 2],
        [3, 1, 2],
    ]
)

token_embedding_table = nn.Embedding(vocab_size, n_embd)
position_embedding_table = nn.Embedding(block_size, n_embd)
block = Block(n_embd, num_heads, block_size)
final_layer_norm = nn.LayerNorm(n_embd)
lm_head = nn.Linear(n_embd, vocab_size)

token_embd = token_embedding_table(idx)
position_ids = torch.arange(T)
position_embd = position_embedding_table(position_ids)
x = token_embd + position_embd
x = block(x)
x = final_layer_norm(x)
logits = lm_head(x)

flat_logits = logits.view(B * T, vocab_size)
flat_targets = targets.view(B * T)
loss = F.cross_entropy(flat_logits, flat_targets)

print("logits shape:", logits.shape)
print("targets shape:", targets.shape)
print("flat_logits shape:", flat_logits.shape)
print("flat_targets shape:", flat_targets.shape)
print("loss:", loss)
