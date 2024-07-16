'''
Code to select reference sequence of the specific level
'''
import pandas as pd
from utils.eval_utils import distance
import numpy as np

min_dist = 200
data = pd.read_csv('data/AAV/hard.csv')
sequences = data['sequence'].tolist()
for i in sequences:
    distances = []
    for j in sequences:
        distances.append(distance(i,j))
    mean_dist = np.mean(distances)
    if mean_dist < min_dist:
        refseq = i 
        min_dist = mean_dist
    
print(refseq)