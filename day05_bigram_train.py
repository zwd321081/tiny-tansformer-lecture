import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

x = torch.tensor([
    [1, 0, 2],
    [2, 3, 1],
])

targets = torch.tensor([
    [0, 2, 2],
    [3, 1, 2],
])

B, T = x.shape
vocab_size = 4

token_embedding_table = nn.Embedding(vocab_size, vocab_size)
optimizer = torch.optim.AdamW(token_embedding_table.parameters(), lr=0.1)


def compute_loss():
    logits = token_embedding_table(x)
    flat_logits = logits.view(B * T, vocab_size)
    flat_targets = targets.view(B * T)
    return F.cross_entropy(flat_logits, flat_targets)


for step in range(51):
    loss = compute_loss()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print("step:", step, "loss:", loss.item())
