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


class TinyTransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, num_heads):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.block = Block(n_embd, num_heads, block_size)
        self.final_layer_norm = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_embd = self.token_embedding_table(idx)
        position_ids = torch.arange(T)
        position_embd = self.position_embedding_table(position_ids)
        x = token_embd + position_embd
        x = self.block(x)
        x = self.final_layer_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, vocab_size = logits.shape
            flat_logits = logits.view(B * T, vocab_size)
            flat_targets = targets.view(B * T)
            loss = F.cross_entropy(flat_logits, flat_targets)

        return logits, loss


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

vocab_size = 4
block_size = 8
n_embd = 8
num_heads = 4

model = TinyTransformerLanguageModel(vocab_size, block_size, n_embd, num_heads)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

for step in range(101):
    logits, loss = model(idx, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 20 == 0:
        print("step:", step, "loss:", loss.item())
