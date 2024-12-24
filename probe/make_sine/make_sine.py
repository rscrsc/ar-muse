import random
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import math

device = torch.device('mps')
torch.set_default_dtype(torch.float32)

@torch.no_grad()
def sample(B, n_sample, Nt=737):
    res = torch.zeros((B, n_sample))
    for i in range(B):
        res[i, :] = torch.roll(torch.sin(2*np.pi*torch.arange((n_sample//Nt+1)*Nt)/Nt), random.randint(0, Nt))[-n_sample:]
    return res

context_length = 2048
n_embedding = 8
n_head = 4
n_block = 1
batch_size = 64

class MyNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.val_embed = nn.Linear(1, n_embedding)
        self.transformer = nn.Sequential(*(Block() for _ in range(n_block)))
        self.norm = nn.RMSNorm(n_embedding)
        self.out = nn.Linear(n_embedding, 1)
    
    def forward(self, x):
        B, S = x.shape
        out = self.out(self.norm(self.val_embed(x.view(B, S, 1))))
        return out

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.RMSNorm(n_embedding)
        self.heads = nn.ModuleList([Head() for _ in range(n_head)])
        self.norm2 = nn.RMSNorm(n_embedding)
        self.feedforward = nn.Sequential(
                            nn.Linear(n_embedding, 4 * n_embedding),
                            nn.GELU(),
                            nn.Linear(4 * n_embedding, n_embedding),
                            nn.GELU())
        self.register_buffer('m', torch.zeros(8).geometric_(0.5))

    def forward(self, x):
        x += torch.cat([head(self.norm1(x), m[i]) for i, head in enumerate(self.heads)], dim=-1)
        x += self.feedforward(self.norm2(x))
        return x

class Head(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = n_embedding // n_head
        self.k = nn.Linear(n_embedding, head_size, bias=False)
        self.q = nn.Linear(n_embedding, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))
        self.register_buffer('alibi', torch.tril(torch.arange(context_length).repeat(context_length, 1) - torch.arange(context_length).repeat(context_length, 1).T))
        self.v = nn.Linear(n_embedding, head_size, bias=False)

    def forward(self, x, m):
        B, S, C = x.shape
        key = self.k(x)
        query = self.q(x)
        att = key @ query.transpose(-2, -1) * (n_embedding ** (-0.5))
        att = att.masked_fill(self.tril[:S, :S], 0.) +  m * self.alibi[:S, :S]
        value = self.v(x)
        out = att @ value
        return out

@torch.no_grad()
def generate(model, x, max_token):
    B, S = x.shape
    if S >= max_token:
        return x
    buffer = torch.zeros((B, S + max_token), device=device)
    buffer[:,:S] = x
    for i in range(max_token):
        context = buffer[:, :S+i] if S+i <= context_length else buffer[:, S+i-context_length:S+i]
#        [0, ..., S-1, S, ..., S + max_token] : 
#            [0, ... S-1] [S  ..., S + max_token - 1] i == 0
#             0 [1.. S-1   S] ..., S + max_token - 1] i == 1
#        || [0, ..., max_token-1, max_token, ..., S]
        out = model(context)
        buffer[:,S+i] = out[:,-1,:].view(B)
    return buffer

max_batch = 1024*8
eval_batch_size = 32
max_eval_batch = 16

model = MyNN().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01, fused=True)
criterion = nn.MSELoss()
for batch in range(max_batch):
    sample_t = sample(batch_size, context_length + 1)
    X_t = sample_t[:, :context_length]
    y_t = sample_t[:, 1:context_length+1].view(batch_size, context_length, 1)
    X_t, y_t = X_t.to(device), y_t.to(device)
    model.train()
    optimizer.zero_grad()
    y_p = model(X_t)
    loss = criterion(y_p, y_t)
    loss.backward()
    optimizer.step()
    model.eval()
    if batch % 128 == 0:
        losses = torch.zeros(max_eval_batch)
        for eval_batch in range(max_eval_batch):
            sample_e = sample(eval_batch_size, context_length + 1)
            X_e = sample_e[:, :context_length]
            y_e = sample_e[:, 1:context_length+1].view(eval_batch_size, context_length, 1)

            X_e, y_e = X_e.to(device), y_e.to(device)
            y_pe = model(X_e)
            losses[eval_batch] = criterion(y_pe, y_e)
        print(f"Batch {batch}, loss = {losses.mean()}")

max_gen = 4096
seed = sample(1, 2)
gen = generate(model, seed, max_gen).detach().tolist()
import matplotlib.pyplot as plt
plt.plot(range(max_gen+seed.shape[-1]), gen[0])
plt.show()
