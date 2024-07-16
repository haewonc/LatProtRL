from typing import Tuple
import torch
import cma
import numpy as np
from baseline.explorer import Explorer
from utils.constants import AAS
from tqdm import tqdm 

class CMAES_Latent(Explorer):
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
        lat_model,
        length,
        device,
        starting_sequences,
        rounds,
        m_decode = 8,
        model_calls_per_round: int = 2048,
		oracle_calls_per_round: int = 256,
  		topk_per_round: int = 128,
        population_size: int = 16,
        initial_variance: float = 0.2,
    ):
        super().__init__(length, starting_sequences, rounds, oracle_calls_per_round, topk_per_round, device)
        
        self.model = model
        self.lat_model = lat_model
        self.m_decode = m_decode
        self.alphabet = AAS
        self.population_size = population_size
        self.max_iter = model_calls_per_round // population_size
        self.initial_variance = initial_variance
        self.round = 0

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
            with torch.no_grad():
                seq = self.lat_model.decode(torch.tensor(soln).float().unsqueeze(0).to(self.device), to_seq=True, template=top_seq, topk=self.m_decode)
            
            if seq in sequences:
                return sequences[seq]
            if seq in measured_sequence_dict:
                return measured_sequence_dict[seq]

            return self.model.get_fitness([seq]).item()

        # Starting solution gives equal weight to all residues at all positions
        with torch.no_grad():
            x0 = self.lat_model.encode(top_seq).cpu().view(-1).detach().numpy()
        opts = {"popsize": self.population_size, "verbose": -9, "verb_log": 0}
        es = cma.CMAEvolutionStrategy(x0, np.sqrt(self.initial_variance), opts)

        for _ in tqdm(range(self.max_iter), desc='iterations'):
            solutions, fitnesses = es.ask_and_eval(objective_function)
            # `tell` updates model parameters
            es.tell(solutions, fitnesses)

            # Store scores of generated sequences
            with torch.no_grad():
                for soln, f in zip(solutions, fitnesses):
                    sequences[self.lat_model.decode(torch.tensor(soln).float().unsqueeze(0).to(self.device), to_seq=True, template=top_seq, topk=self.m_decode)] = f

        # We propose the top `self.sequences_batch_size` new sequences we have generated
        new_seqs = np.array(list(sequences.keys()))
        # Negate `objective_function` scores
        preds = np.array(list(sequences.values()))
        sorted_order = np.argsort(preds)[: -self.oracle_calls_per_round : -1]

        return new_seqs[sorted_order], preds[sorted_order]