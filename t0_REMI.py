# Failed - CELoss >= 2.41, never drop

import torch
from torch.utils.tensorboard import SummaryWriter
import random
import string
import math
import datetime
import re
from miditok import REMI, TokenizerConfig  # here we choose to use REMI
from miditok.pytorch_data import DatasetMIDI
from pathlib import Path

TOKENIZER_PARAMS = {
    "pitch_range": (0, 127),
    "beat_res": {(0, 4): 8, (4, 12): 4},
    "num_velocities": 64,
    "special_tokens": ["PAD", "BOS", "EOS", "MASK"],
    "use_chords": True,
    "use_rests": False,
    "use_tempos": False,
    "use_time_signatures": True,
    "use_programs": False,
}
config = TokenizerConfig(**TOKENIZER_PARAMS)

tokenizer = REMI(config)
midi_paths = list(Path("dataset/midi/adjust_tempo").glob("*.mid"))
dataset_dict = DatasetMIDI(
    files_paths=midi_paths,
    tokenizer=tokenizer,
    max_seq_len=1024,
    bos_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer["BOS_None"],
)

dataset = []
for data in dataset_dict:
    dataset.append(data['input_ids'])

#torch.autograd.set_detect_anomaly(True)
device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
#device = torch.device('cpu')
#torch.manual_seed(42)
torch.set_default_dtype(torch.float32)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class Logger():
    def __init__(self, model_name):
        time_str = str(datetime.datetime.now())
        time_str = re.sub(r'[-: _.]', '-', time_str)
        self.writer = SummaryWriter(log_dir=f'tf-logs/{model_name}_{time_str}')

logger = Logger('AR-Muse')

context_length = 1024       # S = context_length  = max(len(sample) for sample in dataset) - 1
n_embedding = 64 # C = n_channel = n_embedding
head_size = 16        # h = head_size
n_block = 8  # num of transformer blocks
batch_size = 16       # B = batch_size
vocabulary_size = tokenizer.vocab_size
repeating_penalty = 1.0   # repeating penalty when inferencing
zero_penalty = 2.5

dataset_cnt = len(dataset)
trainset_cnt = math.floor(dataset_cnt * 0.9)
testset_cnt = dataset_cnt - trainset_cnt
X_train = torch.zeros(trainset_cnt, context_length, device=device, dtype=torch.int64)
y_train = torch.zeros(trainset_cnt, context_length, device=device, dtype=torch.int64)
X_test = torch.zeros(testset_cnt, context_length, device=device, dtype=torch.int64)
y_test = torch.zeros(testset_cnt, context_length, device=device, dtype=torch.int64)
for i, sample in enumerate(dataset):
    if i < trainset_cnt:
        X_train[i, :len(sample)-1] = sample[:-1]
        y_train[i, :len(sample)-1] = sample[1:]
        y_train[i, len(sample)-1:] = -1    # -1 means target is unused and will be ignored
    else:
        X_test[i-trainset_cnt, :len(sample)-1] = sample[:-1]
        y_test[i-trainset_cnt, :len(sample)-1] = sample[1:]
        y_test[i-trainset_cnt, len(sample)-1:] = -1    # -1 means target is unused and will be ignored 

class MyNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed = torch.nn.Embedding(vocabulary_size, n_embedding)  #  (*) => (*, C)
        self.pos_embed = torch.nn.Embedding(context_length, n_embedding)
        self.transformers = torch.nn.Sequential(
                              *[Block() for _ in range(n_block)])
        self.norm = torch.nn.LayerNorm(n_embedding)
        self.outlayer = torch.nn.Linear(n_embedding, vocabulary_size)

    def forward(self, x, target=None):
        # x.shape == (B, S)
        B, S = x.shape
        tok_emb = self.token_embed(x)                          # (B, S, C)
        pos_emb = self.pos_embed(torch.arange(S, device=device))# (S, C) & broadcast
        out = tok_emb + pos_emb                           # (B, S, C)
        out = self.transformers(out)                      # (B, S, C)
        logits = self.outlayer(self.norm(out))            # (B, S, V)
        if target == None:
            loss = None
        else:
            logits = logits.view(B*S, vocabulary_size)
            target = target.reshape(B*S)
            loss = torch.nn.functional.cross_entropy(logits, target, ignore_index=-1)
        return logits, loss

class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        n_head = n_embedding // head_size
        self.multihead = torch.nn.ModuleList(
                           [Head() for _ in range(n_head)])  # (C//h)*(C, h)
        self.linear = torch.nn.Linear(n_embedding, n_embedding)
        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(n_embedding, 4 * n_embedding),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * n_embedding, n_embedding),
            torch.nn.ReLU(),
        )
        self.norm1 = torch.nn.LayerNorm(n_embedding)
        self.norm2 = torch.nn.LayerNorm(n_embedding)

    def forward(self, x):
        # x.shape == (B, S, C)
        x = x + self.linear(torch.cat([head(self.norm1(x)) for head in self.multihead], dim=-1))
        x = x + self.feedforward(self.norm2(x))
        # residual connection
        return x

class Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.q = torch.nn.Linear(n_embedding, head_size, bias=False)
        self.k = torch.nn.Linear(n_embedding, head_size, bias=False)
        self.v = torch.nn.Linear(n_embedding, head_size, bias=False)

        # nn.Module.register_buffer: Module data that is not a parameter
        self.register_buffer('tril_ones', torch.tril(torch.ones(context_length, context_length)))
    
    def forward(self, x):
        # x.shape == (B, S, C)
        B, S, C = x.shape
        x_q = self.q(x)     # (B, S, h)
        x_k = self.k(x)     # (B, S, h)
        wei = x_q @ x_k.transpose(-2, -1) * (C**-0.5)   # (B, S, S)
        # Decoder-only
        wei.masked_fill(self.tril_ones[:S, :S]==0, -torch.inf)
        wei = torch.softmax(wei, dim=-1)
        x_v = self.v(x)     # (B, S, h)
        out = wei @ x_v     # (B, S, h)
        return out


@torch.no_grad()
def inference(model, x, max_gen_token=context_length-1):
    for i in range(1, max_gen_token):
        logits, loss = model(x.view(1,-1))
        logits[0,-1,x[i-1]] /= repeating_penalty   # repeating penalty
        logits[0,-1,0] /= zero_penalty
        next_prob = torch.nn.functional.softmax(logits[0,-1,:], dim=-1)   # (V)
        token_next = torch.multinomial(next_prob, 1).view(1)              # (1)
        x[i] = token_next
        if token_next.item() == 0:
            break
    return x

# train
model = MyNN().to(device)
model = torch.compile(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01, fused=True)
batch_cnt = 1024*64
eval_iter_cnt = 32
eval_batch_size = 16
for batch in range(batch_cnt):
    model.train()
    train_idx = torch.randint(trainset_cnt, size=(batch_size,))
    X_t = X_train[train_idx] 
    y_t = y_train[train_idx]   # X.shape == y.shape == (B, S)
    optimizer.zero_grad()
    logits, loss = model(X_t, y_t)
    loss.backward()
    optimizer.step()
    if batch % 128 == 0:
        with torch.no_grad():
            model.eval()
            losses = torch.zeros(eval_iter_cnt)
            for i in range(eval_iter_cnt):
                test_idx = torch.randint(testset_cnt, size=(eval_batch_size,))
                X_e = X_test[test_idx]
                y_e = y_test[test_idx]   # X.shape == y.shape == (B, S)
                logits, loss = model(X_e, y_e)
                losses[i] = loss.item()
            print(f'Batch {batch}, loss = {losses.mean()}')
            logger.writer.add_scalar("Average Loss/batch", losses.mean(), batch)
            logger.writer.flush()
    if batch % 128 == 0:
        torch.save(model.state_dict(), 'autodl-tmp/armuse_cpts/b' + str(batch) + '.pth')

