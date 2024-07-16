import random
import torch
import numpy as np
from utils.constants import seq_to_one_hot
from utils.eval_utils import diversity
from heapq import nsmallest

class Buffer:
    def __init__(self, data, random: random.Random, buffer_size=128, step_size=50, gamma=0.96):
        self.buffer_size = buffer_size
        self.random = random
        '''
            Buffer item is 4-tuple of (sequence, fitness, score, visits)
            You may ignore score. it is not mentioned in the paper.
            It is legacy code of internal ablation on buffer design.
        '''
        if isinstance(data, list):
            self.start_wt = False
            self.stats = torch.zeros((len(data[0][0]), 20))
            self.pool = self.initialize_pool(data)
        elif isinstance(data, str): 
            self.start_wt = True
            self.wt = data
            self.pool = []
            self.stats = torch.zeros((len(data), 20))
        else:
            raise NotImplementedError("Buffer should be initialized to dataset or WT")
        self.epsilon = 1.0
        self.step = 0
        self.step_size = step_size
        self.gamma = gamma
        self.exploitation = 2 # temperature
        self.trajs = []

    def initialize_pool(self, data):
        data = sorted(data, key=lambda x: -x[1])
        self.original_data = data 
        candidate_pool = data[:self.buffer_size]
        pool = []
        for c in candidate_pool:
            self.stats += seq_to_one_hot(c[0])
            pool.append([c[0], c[1], c[1], 1])
        return pool

    def find_min(self):
        idx = 0
        fit = self.pool[0][1]
        for i, p in enumerate(self.pool):
            if p[1] < fit: 
                idx = i
                fit = p[1]
        return idx
    
    def push(self, traj):
        self.trajs.append(traj)
        
    def describe(self):
        return self.stats / len(self.pool)
    
    def top(self):
        self.step += 1
        if len(self.pool) == self.buffer_size and self.step % self.step_size == 0:
            self.epsilon = max(0.05, self.gamma * self.epsilon)
        if len(self.pool) < self.buffer_size:
            return self.wt, -1
        
        if self.random.random() < self.epsilon:
            visit = np.array([s[3] for s in self.pool])
            prob = 1 / np.sqrt(visit)
            idx = self.random.choices(range(self.buffer_size), weights=prob, k=1)[0]
            return self.pool[idx][0], idx
        else:
            score = np.array([s[1] for s in self.pool])
            score -= score.min()
            score /= score.max()
            score *= self.exploitation
            score = np.exp(score)/sum(np.exp(score))
            idx = self.random.choices(range(self.buffer_size), weights=score, k=1)[0]
            return self.pool[idx][0], idx
        
    def update(self):
        for traj in self.trajs:
            seq, fitness, buffer_idx = traj
            if buffer_idx == -1 and len(self.pool) < self.buffer_size:
                self.stats += seq_to_one_hot(seq)
                self.pool.append([seq, fitness, fitness, 1])
                continue 
            self.pool[buffer_idx][2] += fitness 
            self.pool[buffer_idx][3] += 1 
            if seq not in [s[0] for s in self.pool]:
                self.pool = sorted(self.pool, key=lambda x: x[1])
                for i, s in enumerate(self.pool):
                    if fitness > s[1]:
                        self.stats -= seq_to_one_hot(s[0])
                        self.stats += seq_to_one_hot(seq)
                        self.pool[i] = [seq, fitness, fitness, 1]
                        break
        self.trajs = []
        for i, s in enumerate(self.pool):
            self.pool[i][2] = s[1]
            self.pool[i][3] = 1
    
    def get_performance(self):
        buffer_diversity = diversity([s[0] for s in self.pool])
        buffer_fitness = sum([s[1] for s in self.pool]) / len(self.pool)
        return buffer_fitness, buffer_diversity
    
    def propose(self, k):
        def sort_key(x):
            return -x[1], x[0]
        top_k_results = nsmallest(k, self.pool, key=sort_key)
        return top_k_results