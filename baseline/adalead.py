"""Defines the Adalead explorer class."""
import random
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from baseline.explorer import Explorer
from utils.constants import AAS, generate_random_mutant
from tqdm import tqdm

class Adalead(Explorer):
    """
    Adalead explorer.

    Algorithm works as follows:
        Initialize set of top sequences whose fitnesses are at least
            (1 - threshold) of the maximum fitness so far
        While we can still make model queries in this batch
            Recombine top sequences and append to parents
            Rollout from parents and append to mutants.

    """

    def __init__(
        self,
        model,
        length,
        device,
        starting_sequences: List[str],
        rounds: int = 20,
        model_calls_per_round: int = 2048,
        oracle_calls_per_round: int = 256, 
        topk_per_round: int = 128,
        mu: int = 1,
        recomb_rate: float = 0,
        threshold: float = 0.05,
        rho: int = 0,
    ):
        """
        Args:
            mu: Expected number of mutations to the full sequence (mu/L per position).
            recomb_rate: The probability of a crossover at any position in a sequence.
            threshold: At each round only sequences with fitness above
                (1-threshold)*f_max are retained as parents for generating next set of
                sequences.
            rho: The expected number of recombination partners for each recombinant.
            eval_batch_size: For code optimization; size of batches sent to model.

        """

        super().__init__(length, starting_sequences, rounds, oracle_calls_per_round, topk_per_round, device)
        self.model = model
        self.threshold = threshold
        self.recomb_rate = recomb_rate
        self.alphabet = AAS
        self.mu = mu  # number of mutations per *sequence*.
        self.rho = rho
        self.model_calls = 0
        self.model_calls_per_round = model_calls_per_round
        self.eval_batch_size = 20
        self.sequences_batch_size = 100

    def _recombine_population(self, gen):
        # If only one member of population, can't do any recombining
        if len(gen) == 1:
            return gen

        random.shuffle(gen)
        ret = []
        for i in range(0, len(gen) - 1, 2):
            strA = []
            strB = []
            switch = False
            for ind in range(len(gen[i])):
                if random.random() < self.recomb_rate:
                    switch = not switch

                # putting together recombinants
                if switch:
                    strA.append(gen[i][ind])
                    strB.append(gen[i + 1][ind])
                else:
                    strB.append(gen[i][ind])
                    strA.append(gen[i + 1][ind])

            ret.append("".join(strA))
            ret.append("".join(strB))
        return ret

    def propose_sequences(
        self
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Propose top `sequences_batch_size` sequences for evaluation."""
        measured_sequence_set = set(self.history["sequence"])

        # Get all sequences within `self.threshold` percentile of the top_fitness
        top_fitness = self.history["true_score"].max()
        top_inds = self.history["true_score"] >= top_fitness * (
            1 - np.sign(top_fitness) * self.threshold
        )
        
        parents = np.resize(
            self.history["sequence"][top_inds].to_numpy(),
            self.sequences_batch_size,
        )
        
        if self.last_round == 0:
            parents = np.resize(
                self.history["sequence"][top_inds].to_numpy(),
                self.oracle_calls_per_round,
            )
            sequences = []
            for i in range(self.oracle_calls_per_round):
                sequences.append(generate_random_mutant(parents[i], mu= 2 / self.length))
            preds = self.model.get_fitness(sequences)
            return np.array(sequences), np.array(preds)
                
        sequences = {}
        self.model_calls = 0
        with tqdm(total=self.model_calls_per_round) as pbar:
            while self.model_calls < self.model_calls_per_round:
                # generate recombinant mutants
                for i in range(self.rho):
                    parents = self._recombine_population(parents)
                    
                for i in range(0, len(parents), self.eval_batch_size):
                    # Here we do rollouts from each parent (root of rollout tree)
                    roots = parents[i : i + self.eval_batch_size]
                    root_fitnesses = self.model.get_fitness(roots)
                    self.model_calls += self.eval_batch_size
                    pbar.update(self.eval_batch_size)
                    
                    nodes = list(enumerate(roots))

                    while (
                        len(nodes) > 0
                        and self.model_calls + self.eval_batch_size
                        < self.model_calls_per_round
                    ):
                        child_idxs = []
                        children = []
                        
                        while len(children) < len(nodes):
                            idx, node = nodes[len(children) - 1]

                            child = generate_random_mutant(
                                node,
                                self.mu * 1 / len(node)
                            )

                            # Stop when we generate new child that has never been seen
                            # before
                            if (
                                child not in measured_sequence_set
                                and child not in sequences
                            ):
                                child_idxs.append(idx)
                                children.append(child)

                        # Stop the rollout once the child has worse predicted
                        # fitness than the root of the rollout tree.
                        # Otherwise, set node = child and add child to the list
                        # of sequences to propose.
                        fitnesses = self.model.get_fitness(children)
                        self.model_calls += 1
                        pbar.update(1)
                        sequences.update(zip(children, fitnesses))

                        nodes = []
                        for idx, child, fitness in zip(child_idxs, children, fitnesses):
                            if fitness >= root_fitnesses[idx]:
                                nodes.append((idx, child))

        if len(sequences) == 0:
            raise ValueError(
                "No sequences generated. If `model_queries_per_batch` is small, try "
                "making `eval_batch_size` smaller"
            )

        # We propose the top `self.sequences_batch_size` new sequences we have generated
        new_seqs = np.array(list(sequences.keys()))
        preds = np.array(list(sequences.values()))
        sorted_order = np.argsort(preds)[: -self.oracle_calls_per_round : -1]

        return new_seqs[sorted_order], preds[sorted_order]
