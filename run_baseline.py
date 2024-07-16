import os
import random 
import torch
import warnings
import argparse
import numpy as np 
import pandas as pd
from baseline.adalead import Adalead
from baseline.adalead_latent import Adalead_Latent
from baseline.bo import BayesianOptimization
from baseline.pex import ProximalExploration
from baseline.cmaes import CMAES
from baseline.cmaes_latent import CMAES_Latent
from baseline.insilico import InSilicoLandscape, Model, Ensemble
from config import create_base, config_rep
from net.seq_lm import VED
from utils.constants import REFSEQ
warnings.simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--protein', type=str, choices=['GFP', 'AAV'], required=True)
parser.add_argument('--level', type=str, choices=['hard', 'medium'], required=True)
parser.add_argument('--alg', type=str, required=True)
parser.add_argument('--device', type=str, required=True)
parser.add_argument('--seed', type=int, default=0, help='Random seed and the run index for the log file name.')
parser.add_argument('--rounds', type=int, default=15)
args = parser.parse_args()

save_name = '{}_{}_{}.csv'.format(args.protein, args.level, args.alg)
cfg = create_base(args)
landscape = InSilicoLandscape(cfg)

starting_sequences = pd.read_csv(f'data/{args.protein}/{args.level}.csv')
starting_sequences.rename(columns={'target': 'true_score'}, inplace=True)
starting_sequences = starting_sequences[['sequence', 'true_score']]
starting_sequences['true_score'] = (starting_sequences['true_score'] - cfg.min_fitness)/(cfg.max_fitness - cfg.min_fitness)

alg = args.alg.lower()
if not os.path.exists(alg):
    os.mkdir(alg)
    
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)    

if alg == 'bo':        
    cnn = Ensemble(N=10, epochs=10, device=args.device)
    explorer = BayesianOptimization(
        model=cnn,
        length=cfg.length,
        starting_sequences=starting_sequences,
        oracle_calls_per_round=256,
        topk_per_round=128,
        rounds=args.rounds,
        device=args.device,
        method="UCB"
    )
elif alg == 'pex':
    cnn = Model(epochs=10, device=args.device)
    explorer = ProximalExploration(
        model=cnn,
        length=cfg.length,
        device=args.device,
        wt_sequence=REFSEQ[args.protein][args.level],
        starting_sequences=starting_sequences,
        rounds=args.rounds,
    )
elif alg == 'adalead':
    cnn = Model(epochs=10, device=args.device)
    explorer = Adalead(
        model=cnn,
        length=cfg.length,
        device=args.device,
        starting_sequences=starting_sequences,
        rounds=args.rounds,
        topk_per_round=128,
        oracle_calls_per_round=256,
        threshold=0.05
    )
elif alg == 'cmaes':
    cnn = Model(epochs=10, device=args.device)
    explorer = CMAES(
        model=cnn,
        length=cfg.length,
        device=args.device,
        starting_sequences=starting_sequences,
        rounds=args.rounds,
        topk_per_round=128,
        oracle_calls_per_round=256,
    )
elif alg == 'cmaes_latent':
    cnn = Model(epochs=10, device=args.device)
    cfg = config_rep(args.device, args.protein, args.level)
    lat_model = VED(cfg, pretrained=f'saved/{args.protein}_{args.level}_LM.pt').to(args.device)
    lat_model.set_wt_tokens(REFSEQ[args.protein][args.level])
    explorer = CMAES_Latent(
        model=cnn,
        lat_model=lat_model,
        length=cfg.length,
        device=args.device,
        m_decode=8 if args.protein == 'AAV' else 12,
        starting_sequences=starting_sequences,
        rounds=args.rounds,
        topk_per_round=128,
        oracle_calls_per_round=256,
    )
elif alg == 'ada_latent':
    cnn = Model(epochs=10, device=args.device)
    cfg = config_rep(args.device, args.protein, args.level)
    lat_model = VED(cfg, pretrained=f'saved/{args.protein}_{args.level}_LM.pt').to(args.device)
    lat_model.set_wt_tokens(REFSEQ[args.protein][args.level])
    explorer = Adalead_Latent(
        model=cnn,
        lat_model=lat_model,
        m_decode=8 if args.protein == 'AAV' else 12,
        delta=0.1 if args.protein == 'AAV' else 0.3,
        length=cfg.length,
        device=args.device,
        starting_sequences=starting_sequences,
        rounds=args.rounds,
        topk_per_round=128,
        oracle_calls_per_round=256,
        threshold=0.05
    )
else:
    raise NotImplementedError()

results = explorer.run(landscape)
results.to_csv(f'{alg}/{args.protein}_{args.level}_{args.seed}_summary.csv',index=False)
explorer.history.to_csv(f'{alg}/{args.protein}_{args.level}_{args.seed}.csv',index=False)