import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.constants import seq_to_one_hot
from typing import List, Tuple, Sequence, Any

def pad_sequences(sequences: Sequence, pad_len, constant_value=0, dtype=None) -> np.ndarray:
    batch_size = len(sequences)
    shape = [batch_size, pad_len]

    if dtype is None:
        dtype = sequences[0].dtype

    if isinstance(sequences[0], np.ndarray):
        array = np.full(shape, constant_value, dtype=dtype)
    elif isinstance(sequences[0], torch.Tensor):
        array = torch.full(shape, constant_value, dtype=dtype)

    for arr, seq in zip(array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq

    return array

class SequenceDataset(Dataset):
    def __init__(self, data, wt, alphabet, in_memory: bool = False):
        self.data = data
        self.wt = wt
        self.batch_converter = alphabet.get_batch_converter()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        sequence = self.data[index]
        mask = [1 if i!=j else 0 for i,j in zip(list(sequence), list(self.wt))]
        mask = [0] + mask + [0]
        return sequence, torch.tensor(mask)

    def collate_fn(self, batch: List[Tuple[Any, ...]]):
        input_ids, mut_masks = tuple(zip(*batch))
        _, _, batch_tokens = self.batch_converter(list(zip(["protein" for _ in range(len(input_ids))], input_ids)))
        return batch_tokens, torch.stack(mut_masks, dim=0)