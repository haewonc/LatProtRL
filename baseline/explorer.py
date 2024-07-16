"""Defines abstract base explorer class."""
import abc
import os
import time
import warnings
from datetime import datetime
from typing import Dict, Optional, Tuple, List
import numpy as np
import pandas as pd
import tqdm


class Explorer(abc.ABC):
    def __init__(
        self,
        length,
        starting_dataset,
        rounds: int,
        oracle_calls_per_round: int, 
        topk_per_round: int,
        device: str,
    ):
        self.length = length 
        
        self.rounds = rounds
        self.oracle_calls_per_round = oracle_calls_per_round
        self.topk_per_round = topk_per_round
        self.device = device
        
        self.starting_dataset = starting_dataset
        seqs, true_score = starting_dataset['sequence'].tolist(), starting_dataset['true_score'].tolist()
        self.starting_sequences = seqs
        self.history = pd.DataFrame(zip([0] * len(seqs), seqs, [0.0] * len(seqs), true_score), columns=['round', 'sequence', 'model_score', 'true_score'])
        self.last_round = 0
    
    def train_model(self, sequences_data, verbose=False):
        self.model.train(sequences_data, verbose)
    
    @abc.abstractmethod
    def propose_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def run(
        self, landscape, verbose: bool = True
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Run the exporer.

        Args:
            landscape: Ground truth fitness landscape.
            verbose: Whether to print output or not.

        """
        # Initial sequences and their scores
        sequences_data = self.starting_dataset
        log = []
        
        print(
            f"round: {0}, top: {sequences_data['true_score'].max()}"
        )
        
        # For each round, train model on available data, propose sequences,
        # measure them on the true landscape, add to available data, and repeat.
        for r in range(1, self.rounds + 1):
            self.train_model(sequences_data, verbose=False)
            
            round_start_time = time.time()
            seqs, preds = self.propose_sequences()
            true_score, fitness, diversity, novelty = landscape.evaluate(seqs, self.starting_sequences, topk=self.topk_per_round)
            
            sequences_data = pd.DataFrame(zip(seqs, true_score), columns=['sequence', 'true_score'])
            round_history = pd.DataFrame(zip([r] * len(seqs), seqs, preds, true_score), columns=['round', 'sequence', 'model_score', 'true_score'])
            log.append([r, true_score.max(), fitness, diversity, novelty])
            self.history = pd.concat([self.history, round_history], axis=0)
            
            print(
                f"round: {r}, top: {true_score.max()}, fit: {fitness}, div: {diversity}, nov: {novelty}",
                f"time: {time.time() - round_start_time:02f}s"
            )
            self.last_round += 1
            
        return pd.DataFrame(log, columns=['round', 'top', 'fitness', 'diversity', 'novelty'])
