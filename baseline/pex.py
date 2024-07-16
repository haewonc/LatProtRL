import random
import numpy as np
from utils.eval_utils import distance
from utils.constants import random_mutation, AAS, generate_random_mutant
from baseline.explorer import Explorer

class ProximalExploration(Explorer):
    """
        Proximal Exploration (PEX)
    """
    
    def __init__(self, 
                 model, 
                 length,
                 device, 
                 wt_sequence,
                 starting_sequences,
                 rounds,
                 model_calls_per_round: int = 2048,
                 oracle_calls_per_round: int = 256, 
                 batch_size: int = 64,
                 topk_per_round: int = 128,
                 num_random_mutations: int = 2,
                 frontier_neighbor_size: int = 5):
        super().__init__(length, starting_sequences, rounds, oracle_calls_per_round, topk_per_round, device,)
        self.model = model
        self.alphabet = AAS
        self.model_calls_per_round = model_calls_per_round
        self.batch_size = batch_size
        self.num_random_mutations = num_random_mutations
        self.frontier_neighbor_size = frontier_neighbor_size
        self.wt_sequence = wt_sequence

    def propose_sequences(self):
        measured_sequence_set = set(self.history["sequence"])
        
        if self.last_round == 0:
            sequences = []
            for i in range(self.oracle_calls_per_round):
                sequences.append(generate_random_mutant(self.wt_sequence, mu= 2 / self.length))
            preds = self.model.get_fitness(sequences)
            return np.array(sequences), np.array(preds)
        
        # Arrange measured sequences by the distance to the wild type.
        measured_sequence_dict = {}
        for _, data in self.history.iterrows():
            distance_to_wt = distance(data['sequence'], self.wt_sequence)
            if distance_to_wt not in measured_sequence_dict.keys():
                measured_sequence_dict[distance_to_wt] = []
            measured_sequence_dict[distance_to_wt].append(data)
        
        # Highlight measured sequences near the proximal frontier.
        frontier_neighbors, frontier_height = [], -np.inf
        for distance_to_wt in sorted(measured_sequence_dict.keys()):
            data_list = measured_sequence_dict[distance_to_wt]
            data_list.sort(reverse=True, key=lambda x:x['true_score'])
            for data in data_list[:self.frontier_neighbor_size]:
                if data['true_score'] > frontier_height:
                    frontier_neighbors.append(data)
            frontier_height = max(frontier_height, data_list[0]['true_score'])

        # Construct the candiate pool by randomly mutating the sequences. (line 2 of Algorithm 2 in the paper)
        # An implementation heuristics: only mutating sequences near the proximal frontier.
        candidate_pool = []
        while len(candidate_pool) < self.model_calls_per_round:
            candidate_sequence = random_mutation(random.choice(frontier_neighbors)['sequence'], self.num_random_mutations)
            if candidate_sequence not in measured_sequence_set:
                candidate_pool.append(candidate_sequence)
                measured_sequence_set.add(candidate_sequence)
        
        # Arrange the candidate pool by the distance to the wild type.
        candidate_pool_dict = {}
        for i in range(0, len(candidate_pool), self.batch_size):
            candidate_batch =  candidate_pool[i:i+self.batch_size]
            model_scores = self.model.get_fitness(candidate_batch)
            for candidate, model_score in zip(candidate_batch, model_scores):
                distance_to_wt = distance(candidate, self.wt_sequence)
                if distance_to_wt not in candidate_pool_dict.keys():
                    candidate_pool_dict[distance_to_wt] = []
                candidate_pool_dict[distance_to_wt].append(dict(sequence=candidate, model_score=model_score))
        for distance_to_wt in sorted(candidate_pool_dict.keys()):
            candidate_pool_dict[distance_to_wt].sort(reverse=True, key=lambda x:x['model_score'])
        
        # Construct the query batch by iteratively extracting the proximal frontier. 
        query_batch = []
        query_model_preds = []
        while len(query_batch) < self.oracle_calls_per_round:
            # Compute the proximal frontier by Andrew's monotone chain convex hull algorithm. (line 5 of Algorithm 2 in the paper)
            # https://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/Convex_hull/Monotone_chain
            stack = []
            for distance_to_wt in sorted(candidate_pool_dict.keys()):
                if len(candidate_pool_dict[distance_to_wt])>0:
                    data = candidate_pool_dict[distance_to_wt][0]
                    new_point = np.array([distance_to_wt, data['model_score']])
                    def check_convex_hull(point_1, point_2, point_3):
                        return np.cross(point_2-point_1, point_3-point_1) <= 0
                    while len(stack)>1 and not check_convex_hull(stack[-2], stack[-1], new_point):
                        stack.pop(-1)
                    stack.append(new_point)
            while len(stack)>=2 and stack[-1][1] < stack[-2][1]:
                stack.pop(-1)
            
            # Update query batch and candidate pool. (line 6 of Algorithm 2 in the paper)
            for distance_to_wt, model_score in stack:
                if len(query_batch) < self.oracle_calls_per_round:
                    query_batch.append(candidate_pool_dict[distance_to_wt][0]['sequence'])
                    query_model_preds.append(model_score)
                    candidate_pool_dict[distance_to_wt].pop(0)

        return np.array(query_batch), np.array(query_model_preds)