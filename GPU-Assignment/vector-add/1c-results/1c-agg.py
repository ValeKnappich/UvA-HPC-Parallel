import numpy as np
from pprint import pprint

d = {
    64: [69, 68, 75, 75, 74, 76, 68, 67, 76, 75],
    128: [75, 68, 71, 67, 71, 69, 67, 73, 75, 68],
    256: [75, 71, 74, 67, 73, 67, 66, 67, 75, 74],
    512: [72, 69, 73, 68, 73, 75, 74, 75, 74, 66],
    1024: [73, 75, 68, 70, 68, 68, 72, 69, 68, 74]
}

winners = {n: 0 for n in d.keys()}
for vs in zip(*d.values()):
    winners[list(d.keys())[np.argmin(vs)]] += 1

agg = {
    n: {
        "mean": np.mean(l),
        "n_fastest": winners[n]
    }
    for n, l in d.items()
}

pprint(agg)