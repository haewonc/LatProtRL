import torch 
import numpy as np
import torch.nn.functional as F
import random 

# wild-type sequence
WT = {
    'GFP': 'SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK',
    'AAV': 'DEEEIRTTNPVATEQYGSVSTNLQRGNR',
}

# reference sequence
REFSEQ = {
    'AAV': {
        'medium': 'DEEEIRTTNPVATEQYGSVETPDEVGNC',
        'hard': 'DEEEIRTTNPFATEQYGSVEEGECQGDF'
    },
    'GFP': {
        'medium': 'SKGEELFTGVVPILVELDGDVNGHKSSVSGEGEGDATYGKLTLKFICTTGKLPVPRPTLATTLSYGVQCLSRYPDHMRQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVSFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK',
        'hard': 'SKGEELFTGVVPILVELDGDVDGHKFSVSGEGEGDATYGKLTLKSICTTGKLPVPWPALVTTLSYGVQCFSRYPDHMKQHDFFKSAMPVGYVQERTIFLKDDGNYKTRAEVRFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEGGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK'
    }
}

AAS = 'ARNDCQEGHILKMFPSTWYV'
ALPHABET = list('ARNDCQEGHILKMFPSTWYV')

IDXTOAA = {i: ALPHABET[i] for i in range(20)}
AATOIDX = {v: k for k, v in IDXTOAA.items()}

def seq_to_idx(seq):
    return [AATOIDX[aa] for aa in list(seq)]

def seq_to_one_hot(seq):
    return F.one_hot(torch.tensor([AATOIDX[aa] for aa in list(seq)]), 20)

def one_hot_to_seq(t):
    idx = torch.argmax(t, dim=1).tolist()
    return ''.join(IDXTOAA[i] for i in idx)

def generate_random_mutant(sequence, mu) -> str:
    mutant = []
    for s in sequence:
        if random.random() < mu:
            mutant.append(random.choice(ALPHABET))
        else:
            mutant.append(s)
    return "".join(mutant)

def random_mutation(sequence, num_mutations):
    wt_seq = list(sequence)
    for _ in range(num_mutations):
        idx = np.random.randint(len(sequence))
        wt_seq[idx] = ALPHABET[np.random.randint(len(ALPHABET))]
    new_seq = ''.join(wt_seq)
    return new_seq