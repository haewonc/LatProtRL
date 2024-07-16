import numpy as np
import pandas as pd
from config import get_fitness_info
import itertools 
from tqdm import tqdm
from utils.eval_utils import distance
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--protein', type=str, choices=['GFP', 'AAV'], required=True)
parser.add_argument('--level', type=str, choices=['hard', 'medium'], required=True)
parser.add_argument('--alg', type=str, required=True)
args = parser.parse_args()

protein = args.protein 
level = args.level

inits = pd.read_csv(f'data/{protein}/{level}.csv')
inits = inits.sort_values(by='target').iloc[:128]['sequence'].tolist()
highs = pd.read_csv(f'data/{protein}/all.csv')[['sequence', 'target']]
highs = highs[highs['target'] > highs['target'].quantile(q=0.9).item()]
highs = highs['sequence'].tolist()
length, min_fitness, max_fitness = get_fitness_info(protein) 
summary = []

if args.alg == 'ours':
    target = 'target'
else:
    target = 'true_score'
    
for run in tqdm(range(5)):
    ddir = f'{args.alg}/{protein}_{level}_{run}.csv'
    sequences = pd.read_csv(ddir)
    for r in tqdm(range(1, 16)):
        data = sequences[sequences['round']==r]
        data = data.sort_values(by=target,ascending=False).iloc[:128]
        if args.alg == 'ours':
            data[target] = (data[target] - min_fitness) / (max_fitness - min_fitness)
        top_fitness = data.iloc[:16][target].mean().item()
        median_fitness = data[target].median().item()
        seqs = data['sequence'].tolist()
        
        distances = []
        for s1, s2 in itertools.combinations(seqs, 2):
            distances.append(distance(s1, s2))
        diversity = np.median(distances)
        
        distances = []
        for j in seqs:
            dist_j = []
            for i in inits:
                dist_j.append(distance(i,j))
            distances.append(min(dist_j))
        novelty = np.median(distances)
        
        distances = []
        for j in seqs:
            dist_j = []
            for i in highs:
                dist_j.append(distance(i,j))
            distances.append(min(dist_j))
        high = np.median(distances)
        
        instance = [run, r, top_fitness, median_fitness, diversity, novelty, high]
        summary.append(instance)
        
results = pd.DataFrame(summary, columns=['run','round','top fitness', 'median fitness','diversity', 'novelty', 'high'])
results.to_csv(f'summary/{args.alg}/{protein}_{level}_total.csv', index=False)