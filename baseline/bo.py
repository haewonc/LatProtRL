import random
import numpy as np
from bisect import bisect_left
import torch
from utils.eval_utils import distance
from utils.constants import AAS, generate_random_mutant, seq_to_one_hot, one_hot_to_seq
from baseline.explorer import Explorer


class BayesianOptimization(Explorer):
    """
        Bayesian Optimization (BO)
    """
    
    def __init__(self, 
                 model, 
                 length,
                 device, 
                 starting_sequences,
                 rounds,
                 model_calls_per_round: int = 4096,
                 oracle_calls_per_round: int = 256, 
                 topk_per_round: int = 128,
                 method: str = "EI"):
        super().__init__(length, starting_sequences, rounds, oracle_calls_per_round, topk_per_round, device)
        assert method in ["EI", "UCB"]
        self.model = model
        self.alphabet = AAS
        self.model_calls_per_round = model_calls_per_round
        self.batch_size = model_calls_per_round // oracle_calls_per_round
        self.method = method 
        self.best_fitness = 0
        self.initial_uncertainty = None
        
        self.num_actions = 0
        self.state = None
    
    def EI(self, vals):
        """Compute expected improvement."""
        return np.mean([max(val - self.best_fitness, 0) for val in vals])

    @staticmethod
    def UCB(vals):
        """Upper confidence bound."""
        discount = 0.01
        return np.mean(vals) - discount * np.std(vals)

    @staticmethod
    def Thompson_sample(measured_batch):
        """Pick a sequence via Thompson sampling."""
        fitnesses = np.cumsum([np.exp(10 * x[0]) for x in measured_batch])
        fitnesses = fitnesses / fitnesses[-1]
        x = np.random.uniform()
        index = bisect_left(fitnesses, x)
        sequences = [x[1] for x in measured_batch]
        return sequences[index]

    def sample_actions(self):
        """Sample actions resulting in sequences to screen."""
        actions = set()
        pos_changes = []
        for pos in range(self.length):
            pos_changes.append([])
            for res in range(len(self.alphabet)):
                if self.state[pos, res] == 0:
                    pos_changes[pos].append((pos, res))
        
        while len(actions) < self.batch_size:
            action = []
            for pos in range(self.length):
                if np.random.random() < 1 / self.length:
                    pos_tuple = pos_changes[pos][
                        np.random.randint(len(self.alphabet) - 1)
                    ]
                    action.append(pos_tuple)
            if len(action) > 0 and tuple(action) not in actions:
                actions.add(tuple(action))
                
        return list(actions)
    
    @staticmethod
    def construct_mutant_from_sample(
        pwm_sample: np.ndarray, one_hot_base: np.ndarray
    ) -> np.ndarray:
        """Return one hot mutant, a utility function for some explorers."""
        one_hot = np.zeros(one_hot_base.shape)
        one_hot += one_hot_base
        i, j = np.nonzero(pwm_sample)  # this can be problematic for non-positive fitnesses
        one_hot[i, :] = 0
        one_hot[i, j] = 1
        return one_hot


    def pick_action(self, all_measured_seqs):
        state = self.state.copy()
        actions = self.sample_actions()
        actions_to_screen = []
        states_to_screen = []
        for i in range(self.batch_size):
            x = np.zeros((self.length, len(self.alphabet)))
            for action in actions[i]:
                x[action] = 1
            actions_to_screen.append(x)
            state_to_screen = self.construct_mutant_from_sample(x, state)
            states_to_screen.append(one_hot_to_seq(torch.tensor(state_to_screen)))
        model_preds = self.model.get_fitness(states_to_screen)
        method_preds = (
            [self.EI(vals) for vals in model_preds]
            if self.method == "EI"
            else [self.UCB(vals) for vals in model_preds]
        )
        action_ind = np.argmax(method_preds)
        uncertainty = np.std(model_preds[action_ind])
        action = actions_to_screen[action_ind]
        
        new_state_string = states_to_screen[action_ind]
        self.state = seq_to_one_hot(new_state_string).numpy()
        reward = np.mean(model_preds[action_ind])
        if new_state_string not in all_measured_seqs:
            self.best_fitness = max(self.best_fitness, reward)
        self.num_actions += 1
        return uncertainty, new_state_string, reward
    
    def propose_sequences(self):
        
        last_batch = self.history[self.history["round"] == self.last_round]
        last_batch = last_batch.sort_values(by='true_score', ascending=False)
        last_batch_tuples = list(last_batch[['true_score', 'sequence']].itertuples(index=False, name=None))
        sampled_seq = self.Thompson_sample(last_batch_tuples)
        self.state = seq_to_one_hot(sampled_seq).numpy()
        
        self.initial_uncertainty = None
        samples = set()
        all_measured_seqs = set(self.history["sequence"].tolist())
        
        for i in range(self.model_calls_per_round // self.batch_size):
            uncertainty, new_state_string, _ = self.pick_action(all_measured_seqs)
            
            all_measured_seqs.add(new_state_string)
            samples.add(new_state_string)
            
            if self.initial_uncertainty is None:
                self.initial_uncertainty = uncertainty

            if uncertainty > 1.2 * self.initial_uncertainty:
                # reset sequence to starting sequence if we're in territory that's too
                # uncharted
                sampled_seq = self.Thompson_sample(last_batch_tuples)
                self.state = seq_to_one_hot(sampled_seq).numpy()
                self.initial_uncertainty = None
            
        print(f'Random generating {self.oracle_calls_per_round-len(samples)} sequences')
        while len(samples) < self.oracle_calls_per_round:
            random_sequence = generate_random_mutant(
                random.choice(list(samples)), 2 / self.length
            )
            samples.add(random_sequence)
        samples = list(samples)
        preds = self.model.get_fitness(samples).mean(axis=1)
            
        return np.array(samples), preds