import itertools

def distance(s1, s2):
    return sum([1 if i!=j else 0 for i,j in zip(list(s1), list(s2))])

def diversity(seqs):
    divs = []
    for s1, s2 in itertools.combinations(seqs, 2):
        divs.append(distance(s1, s2))
    return sum(divs) / len(divs)

def mean_distance(seq, seqs):
    divs = []
    for s in seqs:
        divs.append(distance(seq, s))
    return sum(divs) / len(divs)