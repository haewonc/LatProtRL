import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List
import torch.nn as nn
import itertools
from tqdm import tqdm
from net.rew import BaseCNN
from torch.optim import Adam
from utils.constants import seq_to_one_hot
from utils.eval_utils import distance 

class TrainDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, idx):
        return seq_to_one_hot(self.data.iloc[idx]['sequence']), torch.tensor(self.data.iloc[idx]['true_score'])
    
    def __len__(self):
        return len(self.data)
    
class SequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
    
    def __getitem__(self, idx):
        return seq_to_one_hot(self.sequences[idx])
    
    def __len__(self):
        return len(self.sequences)
    

class Ensemble:
    def __init__(self, N, epochs, device):
        self.models = [
            Model(epochs, device) for i in range(N)
        ]
        
    def train(self, data, verbose=False):
        for model in self.models:
            model.train(data.sample(frac=.8), verbose)
        
    def get_fitness(self, sequences: List[str]):
        ensembles = []
        for model in self.models:
            ensembles.append(model.get_fitness(sequences))
        return np.stack(ensembles, axis=1)       
    
class Model:
    def __init__(self, epochs, device):
        model = BaseCNN(make_one_hot=False)
        self.model = model.to(device)
        self.epochs = epochs 
        self.device = device
    
    def train(self, data, verbose=False):
        dset = TrainDataset(data)
        dloader = DataLoader(dset, batch_size=128)
        optimizer = Adam(self.model.parameters(), lr=1e-4)
        loss_fn = nn.MSELoss()
        self.model.train()
        for epoch in tqdm(range(1, self.epochs + 1), desc='Training Model'):
            total_loss = 0.0
            for seqs, scores in dloader:
                optimizer.zero_grad()
                seqs, scores = seqs.float().to(self.device), scores.float().to(self.device)
                preds = self.model(seqs)
                loss = loss_fn(preds, scores)
                loss.backward()
                total_loss += loss.item()
                optimizer.step()
        print(f'Model mse {round(total_loss/len(dloader),3)}')
            
    def get_fitness(self, sequences: List[str]):
        dset = SequenceDataset(sequences)
        dloader = DataLoader(dset, batch_size=128)
        scores = []
        self.model.eval()
        for data in dloader:
            data = data.to(self.device)
            with torch.no_grad():
                score = self.model(data).detach().cpu().numpy().astype(float).reshape(-1)
            scores.append(score)

        return np.concatenate(scores)
    
class InSilicoLandscape:
    def __init__(self, cfg):
        self.device = cfg.device
        oracle = BaseCNN(make_one_hot=False)
        oracle_ckpt = torch.load(cfg.rew_pretrained, map_location=self.device)
        if "state_dict" in oracle_ckpt.keys():
            oracle_ckpt = oracle_ckpt["state_dict"]
        oracle.load_state_dict({ k.replace('predictor.',''):v for k,v in oracle_ckpt.items() })
        oracle.eval()
        self.config = cfg
        self.oracle = oracle.to(self.device)
        
    def evaluate(self, sequences, starting_sequences, topk):
        scores = self.get_fitness(sequences)
        indices = np.argsort(scores)[::-1][:topk]
        sequences, targets = sequences[indices], scores[indices]
        fitness = np.median(targets)
        
        distances = []
        for s1, s2 in itertools.combinations(sequences, 2):
            distances.append(distance(s1, s2))
        diversity = np.median(distances)
        
        distances = []
        for j in sequences:
            dist_j = []
            for i in starting_sequences:
                dist_j.append(distance(i,j))
            distances.append(min(dist_j))
        novelty = np.median(distances)

        return scores, fitness, diversity, novelty
    
    def normalize_target(self, target):
        return (target - self.config.min_fitness)/(self.config.max_fitness - self.config.min_fitness)

    def get_fitness(self, sequences: List[str]):
        dset = SequenceDataset(sequences)
        dloader = DataLoader(dset, batch_size=128)
        scores = []
        for data in dloader:
            data = data.to(self.device)
            with torch.no_grad():
                score = self.oracle(data).detach().cpu().numpy().astype(float).reshape(-1)
            scores.append(self.normalize_target(score))

        return np.concatenate(scores)
