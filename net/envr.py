import torch 
import warnings
import numpy as np
import pandas as pd
from gymnasium import Env, spaces 
from utils.constants import REFSEQ, ALPHABET, seq_to_one_hot
from utils.eval_utils import distance
from net.buffers import * 
from net.rew import BaseCNN
from net.seq_lm import VED
from config import * 

warnings.filterwarnings('ignore')

class SingleOpt(Env):
    def __init__(self, config, seed=422):
        super().__init__()
        seq_cfg = create_rep_from_opt(config)
        obs_shape = (seq_cfg.reduce_dim, )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
        self.action_space = spaces.Box(low=-config.action_size, high=config.action_size, shape=obs_shape, dtype=np.float32)

        self.device = seq_cfg.device
        self.protein = config.name
        self.length = config.length
        self.done_cond = config.done_cond
        self.oracle_calls = 0
        self.config = config

        oracle = BaseCNN(make_one_hot=False)
        oracle_ckpt = torch.load(config.rew_pretrained, map_location=self.device)
        if "state_dict" in oracle_ckpt.keys():
            oracle_ckpt = oracle_ckpt["state_dict"]
        oracle.load_state_dict({ k.replace('predictor.',''):v for k,v in oracle_ckpt.items() })
        oracle.eval()
        self.oracle = oracle.to(self.device)

        model = VED(seq_cfg, pretrained=config.seq_pretrained)
        model.eval()
        self.model = model.to(self.device)

        data = pd.read_csv('data/{}/{}.csv'.format(config.name, config.level))[["sequence", "target"]]
        data["target"] = (data["target"] - self.config.min_fitness)/self.config.max_fitness
        self.buffer = Buffer(list(data.itertuples(index=False, name=None)), random=random.Random(seed))
        self.inits = data["sequence"].tolist()
        self.model.set_wt_tokens(REFSEQ[self.protein][self.config.level])
        
        self.ep = 0
        self.steps = 0
        self.total_steps = 0
        self.state = None 
        self.state_seq = None # String 
        self.wt_seq = REFSEQ[self.protein][self.config.level]
        self.init_target = 0
        self.buffer_idx = None

        # logging
        self.reward = 0
        self.done = False
        self.target = 0
        self.best_discovered = 0
        self.init_seq = None
        self.n_mut = 0 
        self.aa = {a:0 for a in ALPHABET}
        self.pos = {a:0 for a in range(self.length)}

    def normalize_target(self, target):
        return (target - self.config.min_fitness)/(self.config.max_fitness - self.config.min_fitness)
    
    def record_mutation(self, s1, s2):
        for pos, (i,j) in enumerate(zip(list(s1), list(s2))):
            if i != j:
                self.aa[j] += 1
                self.pos[pos] += 1
    
    def reset(self, seed=422):
        self.state_seq, self.buffer_idx = self.buffer.top()
        self.init_seq = self.state_seq
        with torch.no_grad():
            state = self.model.encode(self.state_seq)
            self.state = state.cpu().view(-1)
            _, target = self.oracle(seq_to_one_hot(self.state_seq).unsqueeze(0).to(self.device), get_embed=True)
            self.init_target = self.target = target = self.normalize_target(target.cpu().item())
        self.ep += 1
        self.steps = 0
        self.total_steps += 1
        return self.state, {}
    
    def step(self, action):
        self.steps += 1
        self.total_steps += 1
        next_state = (torch.tensor(action) + self.state).unsqueeze(0).to(self.device)
         
        with torch.no_grad():
            next_seq = self.model.decode(next_state, to_seq=True, template=self.state_seq, topk=self.config.topk)
            self.record_mutation(self.state_seq, next_seq)
            self.step_mut = step_mut = distance(self.state_seq, next_seq)
            self.n_mut = distance(self.wt_seq, next_seq)
        
        self.done = done = step_mut > self.done_cond.step_mut or self.steps > self.done_cond.max_steps or self.n_mut > self.done_cond.max_mutation 
        called = False
        
        if step_mut <= self.done_cond.step_mut:
            if self.config.not_sparse or done:
                self.oracle_calls += 1
                with torch.no_grad():
                    _, target = self.oracle(seq_to_one_hot(next_seq).unsqueeze(0).to(self.device), get_embed=True)
                target = self.normalize_target(target.cpu().item())
                self.reward = target
                self.buffer.push((next_seq, target, self.buffer_idx))
                if target > self.best_discovered:
                    self.best_discovered = target
                self.target = target
                called = True
            else:
                self.reward = 0
        else:
            target = 0
            self.reward = -1
        
        self.state = next_state.cpu().numpy()[0]
        self.state_seq = next_seq
            
        if done:
            info = {
                'candidates': self.state_seq, 
                'fitness': self.target,
                'init_seq': self.init_seq,
                'n_mut': self.n_mut,
                'aa': self.aa,
                'pos': self.pos,
                'called': called
            }
            return self.state, self.reward, done, False, info
        return self.state, self.reward, done, False, {}