'''
Evaluate the optimized sequences from log file and save as the summary format.

1. Copy the run subdirectory in the results directory to root_dir
2. Each subdirectory should be formatted as [PROTEIN]_[LEVEL]_[RUN INDEX]  
3. In our paper we ran 5 runs with different seed, which means [RUN INDEX] ranges from 0 to 4
'''
import os
import torch 
import argparse
import numpy as np
from tqdm import tqdm 
import pandas as pd
from utils.constants import seq_to_one_hot
from net.rew import BaseCNN 

root_dir = 'summary/ours_raw' 
R = 15

parser = argparse.ArgumentParser()
parser.add_argument('--protein', type=str)
parser.add_argument('--level', type=str)
parser.add_argument('--device', type=str)
args = parser.parse_args()

protein = args.protein 
level = args.level
device = args.device

rounds = []
sequences = []
targets = []

if not os.path.exists('summary/ours'):
    os.mkdir('summary/ours')

oracle = BaseCNN(make_one_hot=False)
oracle_ckpt = torch.load(f'ckpt/{protein}/oracle.ckpt', map_location=device)
if "state_dict" in oracle_ckpt.keys():
    oracle_ckpt = oracle_ckpt["state_dict"]
oracle.load_state_dict({ k.replace('predictor.',''):v for k,v in oracle_ckpt.items() })
oracle.eval()
oracle = oracle.to(device)

for r in range(5):
    base_dir = f'{root_dir}/{protein}_{level}_{r}'
    for i in tqdm(range(1,R+1)):
        seqs = np.load(base_dir + '/' + f'{i+1}.npy')
        sequences.extend(seqs)
        tensors = torch.stack([seq_to_one_hot(seq) for seq in seqs], dim=0).to(device)
        with torch.no_grad():
            targets.extend(oracle(tensors).cpu().numpy())
        rounds.extend([i] * len(seqs))
        
    pd.DataFrame(list(zip(rounds, sequences, targets)), columns=['round', 'sequence', 'target']).to_csv(f'summary/ours/{protein}_{level}_{r}.csv', index=False)