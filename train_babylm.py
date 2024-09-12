from MEGABYTE_pytorch import MEGABYTE

import random
import tqdm
import gzip
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

# constants

NUM_BATCHES = int(2e5)
BATCH_SIZE = 30
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 2e-4
VALIDATE_EVERY  = 100
GENERATE_EVERY  = 500
PRIME_LEN = 100
SEQ_LEN = 8192
NAS_DATA_DIR = '/mnt/nas/divsinghal/tokenization/data'
# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# instantiate GPT-like decoder model

model = MEGABYTE(
    num_tokens = 256,
    dim = (64, 64),  # dimension of each attention head
    depth = (12,8),   # no of layers in global and local
    max_seq_len = (768, 768),   
    flash_attn = True,
    heads = 12,  # number of heads
).cuda()

# prepare enwik8 data

# with gzip.open('./data/enwik8.gz') as file:
#     x = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
#     train_x, valid_x = np.split(x, [int(90e6)])
#     print("X: ", x, len(x))
#     print("TRAIN: ", train_x, len(train_x))
#     print("VALID: ", valid_x, len(valid_x))
#     data_train, data_val = map(torch.from_numpy, (train_x, valid_x))
with open('./data/enwik8', 'rb') as file:  # Change here
    x = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
    train_x, valid_x = np.split(x, [int(90e6)])
    # x_char = [chr(val) for val in x]
    # print("X: ", x, len(x), ''.join(x_char)[:300])
    # print("TRAIN: ", train_x, len(train_x))
    # print("VALID: ", valid_x, len(valid_x))
    data_train, data_val = map(torch.from_numpy, (train_x, valid_x)) 

# needs no preprocessing
file = open(NAS_DATA_DIR + "/train_10M/bnc_spoken.train")
bnc_spokentrain = file.readlines()
bnc_spokentrain = [line.replace("\n", "").strip() for line in bnc_spokentrain]
file.close()

file = open(NAS_DATA_DIR + "/train_10M/childes.train")  # we need to remove the speaker information
childes_train = file.readlines()
childes_train = [line.replace("\n", "").strip() for line in childes_train]
file.close()

childes_train_proc = []
for line in tqdm.tqdm(childes_train):
    if ":" in line:
        line = line.split(":")[1].replace("\n", "").strip()
    elif line != "" and len(line) > 0: line = line.replace("\n", "").strip() 
    else: continue
    childes_train_proc.append(line)

# needs no preprocessing
file = open(NAS_DATA_DIR + "/train_10M/gutenberg.train")
guten_train = file.readlines()
guten_train = [line.replace("\n", "").strip() for line in guten_train]
file.close()

# needs no preprocessing
file = open(NAS_DATA_DIR + "/train_10M/open_subtitles.train")
subtitle_train = file.readlines()
subtitle_train = [line.replace("\n", "").strip() for line in subtitle_train]
file.close()

file = open(NAS_DATA_DIR + "/train_10M/simple_wiki.train")  # remove the == lines
wiki_train = file.readlines()
file.close()

wiki_train_proc = []
for line in tqdm.tqdm(wiki_train):
    if "= = =" in line: continue
    line = line.replace("\n", "").strip()
    wiki_train_proc.append(line)

file = open(NAS_DATA_DIR + "/train_10M/switchboard.train")  # we need to remove the speaker information
sboard_train = file.readlines()
file.close()

sboard_train_proc = []
for line in tqdm.tqdm(sboard_train):
    if ":" in line:
        line = line.split(":")[1].replace("\n", "").strip()
    elif line != "" and len(line) > 0: line = line.replace("\n", "").strip()
    else: continue
    sboard_train_proc.append(line)

all_train = bnc_spokentrain + childes_train_proc + guten_train + subtitle_train + wiki_train_proc + sboard_train_proc
print("Total training sentences are: ", len(all_train))
print(all_train[:10])
# train_dict = {"text": all_train}
# train_dataset = Dataset.from_dict(train_dict)

uint8_all_train = []
newline_char = np.uint8(ord('\n'))
for sentence in all_train:
    uint8_all_train.extend(sentence.encode('utf-8'))
    uint8_all_train.append(newline_char)

uint8_array = np.array(uint8_all_train, dtype=np.uint8)
train_x, valid_x = np.split(uint8_array, [int(0.95 * len(uint8_array))])
data_train, data_val = map(torch.from_numpy, (train_x, valid_x)) 

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start: rand_start + self.seq_len].long()
        return full_seq.cuda()

    def __len__(self):
        return self.data.size(0) // self.seq_len

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset   = TextSamplerDataset(data_val, SEQ_LEN)
train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE))
val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE))

# optimizer

optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas = (0.9, 0.98))

# training
for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()

    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        loss = model(next(train_loader), return_loss = True)
        loss.backward()

    print(f'{i} : training loss: {loss.item()}')
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()
    optim.zero_grad()

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            loss, bpb = model(next(val_loader), return_loss = True, return_bpb=True)
            print(f'{i} : validation loss: {loss}, bpb : {bpb}')

    if i != 0 and i % GENERATE_EVERY == 0:
        model.eval()
        inp = random.choice(val_dataset)[:-1]
        prime_inp = inp[:PRIME_LEN]
        prime = decode_tokens(prime_inp)
        print('Epoch : %d \n\n%s \n\n%s' % (i, prime, '*' * 100))

        sample = model.generate(prime_inp[None, :])
        sample = sample.flatten(1)

        output_str = decode_tokens(sample[0][PRIME_LEN:])
        print(f'{i} : {output_str}')
