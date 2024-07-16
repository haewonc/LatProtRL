import os
import torch 
import argparse
import numpy as np
import torch.nn as nn
import pandas as pd
from net.seq_lm import VED
from utils.constants import REFSEQ, generate_random_mutant
from utils.datasets import SequenceDataset
from torch.utils.data import DataLoader, random_split
import time
from config import *
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--protein', type=str, choices=['GFP', 'AAV'], required=True)
parser.add_argument('--level', type=str, choices=['hard', 'medium'], required=True)
parser.add_argument('--device', type=str, required=True)
parser.add_argument('--batch', type=int, required=True)
parser.add_argument('--augment', type=int, default=4)
parser.add_argument('-r', '--reduce_dim', type=int)
parser.add_argument('-l', '--num_trainable_layers', type=int, default=4)
args = parser.parse_args()

protein = args.protein
level = args.level
device = args.device
batch_size = args.batch
save_name = '{}_{}_{}'.format(protein, level, time.strftime('%d-%H:%M', time.localtime(time.time())))

if not os.path.exists('saved/'):
    os.mkdir('saved')

total_epochs = 32
use_scheduler = True
log = True

config = config_rep(device, protein, level)
config.reduce_dim = args.reduce_dim
config.num_trainable_layers = args.num_trainable_layers
model = VED(config, esm_pretrained='ckpt/esm2_t33_650M_UR50D.pt')
model = model.to(device) 
wt_tokens = model.compose_input([('protein', REFSEQ[protein][args.level])]).cpu()
model.set_wt_tokens(REFSEQ[protein][args.level])

data = pd.read_csv(f'data/{protein}/{level}.csv')
val_size = test_size = int(len(data)* 0.05)

data = data.sample(frac=1) # shuffule
data.iloc[:test_size].to_csv(f'data/{protein}/{level}_test.csv')
data = data.iloc[test_size:]
data = [i for i in data["sequence"]]

def get_augmented(original):
    return [generate_random_mutant(seq, 3 / config.length) for seq in original]

original = data.copy()
for k in range(args.augment):
    data += get_augmented(original)
dataset = SequenceDataset(data, REFSEQ[protein][args.level], model.alphabet)

train_set, val_set = random_split(dataset, [len(dataset)-val_size, val_size])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)

ce_loss = nn.CrossEntropyLoss(reduce=False)
step_size = len(train_loader)

optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3, weight_decay=1e-5)
if use_scheduler:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size*256, gamma=0.5)

if log: 
    wandb.init(project="LatProtRL", name=save_name)
    wandb.watch(models=model)

def epoch_loop(state, epoch, loader):
    global total_epochs
    assert state in ['train', 'test']

    desc = {}
    for key in ['CELoss', 'Acc', 'Mutated_Acc', 'True_Dist_WT', 'Pred_Dist_WT']:
        desc[key] = []

    for idx, data in enumerate(tqdm(loader, desc=f'{state} {epoch}/{total_epochs}')):
        seq, mask = data
        seq, mask = seq.to(device), mask.to(device)
        if state == 'train':
            optimizer.zero_grad()
            pred_seq, repr = model(seq, return_rep=True)
        else:
            with torch.no_grad():
                pred_seq, repr = model(seq, return_rep=True)
        
        batch_size = mask.size(0)
        seq = seq.flatten()
        mask = mask.flatten()
        pred = pred_seq.view(-1, pred_seq.size(-1))
        loss_pos = ce_loss(pred, seq) 
        # upweight the mutated positions wrt reference sequence
        loss_val = (0.1 * torch.sum(loss_pos * mask)/torch.sum(mask) + 0.9 * torch.sum(loss_pos * ~mask)/torch.sum(~mask))/2

        if state == 'train':
            loss_val.backward()
            optimizer.step()
            if use_scheduler:
                scheduler.step()
        
        seq, pred_seq, mask = seq.detach().cpu().numpy(), pred.detach().cpu().numpy(), mask.detach().cpu().numpy()

        pred_seq = np.argmax(pred_seq, axis=1)
        desc["CELoss"].append(loss_val.item())
        desc["Acc"].append(accuracy_score(seq, pred_seq))
        desc["Mutated_Acc"].append(accuracy_score(seq[mask==1], pred_seq[mask==1]))
        desc["True_Dist_WT"].append(mask.sum()/batch_size) 
        desc["Pred_Dist_WT"].append(config.num_tokens-accuracy_score(wt_tokens.repeat(batch_size, 1).flatten().numpy(), pred_seq, normalize=False)/batch_size)

        repr = repr.detach().cpu().numpy()
        
        if idx == 1: # log mean/stdev of representation
            tqdm.write('\t'.join(['%.2f (%.2f)'%(m,s) for m,s in zip(np.mean(repr, axis=0), np.std(repr, axis=0))]))
        
        if state == 'train':
            desc_iter = {}
            for k in desc:
                desc_iter["{}/{}".format(state, k)] = desc[k][-1] 
            if log:
                wandb.log(desc_iter)

    desc_epoch = {}
    for k in desc: 
        total = np.mean(np.array(desc[k]))
        desc_epoch["{}/Total_{}".format(state, k)] = total
    if log:
        wandb.log(desc_epoch)

for epoch in range(total_epochs):
    model.train()
    epoch_loop("train", epoch, train_loader)
    with torch.no_grad():
        model.eval()
        epoch_loop("val", epoch, val_loader)
    
    torch.save(model.state_dict(), 'saved/{}.pt'.format(save_name))
