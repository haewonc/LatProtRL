from typing import Tuple
import torch
import cma
import numpy as np
from baseline.explorer import Explorer
from utils.constants import AAS, seq_to_one_hot, one_hot_to_seq

class CMAES(Explorer):
    """
    An explorer which implements the covariance matrix adaptation evolution
    strategy (CMAES).

    Optimizes a continuous relaxation of the one-hot sequence that we use to
    construct a normal distribution around, sample from, and then argmax to get
    sequences for the objective function.

    http://blog.otoro.net/2017/10/29/visual-evolution-strategies/ is a helpful guide.
    """

    def __init__(
        self,
        model,
        length,
        device,
        starting_sequences,
        rounds,
        model_calls_per_round: int = 4096,
		oracle_calls_per_round: int = 256,
  		topk_per_round: int = 128,
        population_size: int = 16,
        initial_variance: float = 0.2,
    ):
        super().__init__(length, starting_sequences, rounds, oracle_calls_per_round, topk_per_round, device)
        
        self.model = model
        self.alphabet = AAS
        self.population_size = population_size
        self.max_iter = model_calls_per_round // population_size
        self.initial_variance = initial_variance
        self.round = 0

    def _soln_to_string(self, soln):
        x = soln.reshape((self.length, len(self.alphabet)))

        one_hot = np.zeros(x.shape)
        one_hot[np.arange(len(one_hot)), np.argmax(x, axis=1)] = 1

        return one_hot_to_seq(torch.tensor(one_hot))

    def propose_sequences(
        self
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Propose top `sequences_batch_size` sequences for evaluation."""
        measured_sequences = self.history[self.history["round"] == self.last_round]
        measured_sequence_dict = dict(
            zip(measured_sequences["sequence"], measured_sequences["true_score"])
        )

        # Keep track of new sequences generated this round
        top_idx = measured_sequences["true_score"].argmax()
        top_seq = measured_sequences["sequence"].to_numpy()[top_idx]
        top_val = measured_sequences["true_score"].to_numpy()[top_idx]
        sequences = {top_seq: top_val}

        def objective_function(soln):
            seq = self._soln_to_string(soln)

            if seq in sequences:
                return sequences[seq]
            if seq in measured_sequence_dict:
                return measured_sequence_dict[seq]

            return self.model.get_fitness([seq]).item()

        # Starting solution gives equal weight to all residues at all positions
        x0 = seq_to_one_hot(top_seq).flatten()
        opts = {"popsize": self.population_size, "verbose": -9, "verb_log": 0}
        es = cma.CMAEvolutionStrategy(x0, np.sqrt(self.initial_variance), opts)

        for _ in range(self.max_iter):
            solutions, fitnesses = es.ask_and_eval(objective_function)
            # `tell` updates model parameters
            es.tell(solutions, fitnesses)

            # Store scores of generated sequences
            for soln, f in zip(solutions, fitnesses):
                sequences[self._soln_to_string(soln)] = f

        # We propose the top `self.sequences_batch_size` new sequences we have generated
        new_seqs = np.array(list(sequences.keys()))
        # Negate `objective_function` scores
        preds = np.array(list(sequences.values()))
        sorted_order = np.argsort(preds)[: -self.oracle_calls_per_round : -1]

        return new_seqs[sorted_order], preds[sorted_order]