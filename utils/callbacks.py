import numpy as np
import wandb
import json
import warnings
from metric import Evaluator
from stable_baselines3.common.callbacks import BaseCallback

class BufferLoggingCallback(BaseCallback):
  def __init__(self, args, save_dir, pth_dir, verbose=0):
    super().__init__(verbose)
    self.save_dir = save_dir
    self.pth_dir = pth_dir
    self.evaluator = Evaluator(args.name, args.max_fitness, args.min_fitness, args.device) 
    self.rounds = 0
    
  def _on_step(self) -> bool:
    return True
  
  def _on_rollout_end(self) -> bool:
    
    buffer = self.model.env.get_attr('buffer')[0]
    infos = buffer.propose(k=128)
    seqs = [info[0] for info in infos]
    inits = self.model.env.get_attr('inits')[0]
    fit, div, nov, high = self.evaluator.evaluate(seqs, inits)

    log = {
        'eval/round': self.rounds,
        'eval/fitness': fit, 
        'eval/diversity': div,
        'eval/novelty': nov,
        'eval/high': high
    }
    self.rounds += 1
    np.save('{}/{}.npy'.format(self.save_dir, self.rounds), seqs)
    self.model.save(self.pth_dir+'/{}.zip'.format(self.rounds))
    wandb.log(log)
    with open('{}/{}.txt'.format(self.save_dir, self.rounds), 'w') as file:
        json.dump({k: float(v) for k,v in log.items()}, file)
    

class RewardLoggingCallback(BaseCallback):
  def __init__(self, verbose=0):
    super().__init__(verbose)
    self.cumul = 0
  
  def _on_step(self) -> bool:
    # Log reward
    target = self.model.env.get_attr('target')[0]
    reward = self.model.env.get_attr('reward')[0]
    best = self.model.env.get_attr('best_discovered')[0]
  
    self.cumul += reward
    log = {
      'Fitness': target, 
      'Cumulative Reward': self.cumul, 
      'Cumulative Best': best
      }

    if self.model.env.get_attr('done')[0]:
      ep = self.model.env.get_attr('ep')[0]
      log['Mutation from WT'] = self.model.env.get_attr('n_mut')[0]
      log['Step Mutation'] = self.model.env.get_attr('step_mut')[0]
      log['Episode'] = ep   
      log['Oracle Calls'] = self.model.env.get_attr('oracle_calls')[0]
      if ep % 200 == 199:
        aa = self.model.env.get_attr('aa')[0]
        pos = self.model.env.get_attr('pos')[0]
        table_aa = wandb.Table(data=[[label, val] for (label, val) in aa.items()], columns = ["type", "count"])
        log['Mutated AA'] = wandb.plot.bar(table_aa, "type", "count", title="AA")
        table_pos = wandb.Table(data=[[label, val] for (label, val) in pos.items()], columns = ["position", "count"])
        log['Mutated Position'] = wandb.plot.bar(table_pos, "position", "count", title="Position")
        buffer = self.model.env.get_attr('buffer')[0]
        if len(buffer.pool) == buffer.buffer_size:
            fitness, diversity = buffer.get_performance()
            log['buffer/epsilon'] = buffer.epsilon
            log['buffer/fitness'] = fitness
            log['buffer/diversity'] = diversity
    try: 
      wandb.log(log)
    except FileNotFoundError: 
        # sometimes wandb fails to log table
        warnings.warn("tmp dir problem")
        del log['Mutated AA']
        del log['Mutated Position']
        wandb.log(log)
    return True