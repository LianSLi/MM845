"""Statistics for paired comparisons: exact sign test and cluster bootstrap."""
import math
from collections import defaultdict

import numpy as np


def sign_test_p(b, c):
    """Two-sided exact sign test on discordant pairs (b: first method only, c: second only)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def cluster_bootstrap(rows, f, unit, B=4000, seed=0):
    """Mean of f(row) with a 95% percentile bootstrap interval, resampling whole clusters unit(row)."""
    cl = defaultdict(list)
    for r in rows:
        cl[unit(r)].append(f(r))
    keys = list(cl)
    s = np.array([sum(cl[k]) for k in keys], dtype=float)
    n = np.array([len(cl[k]) for k in keys], dtype=float)
    rng = np.random.default_rng(seed)
    est = [s[i].sum() / n[i].sum() for i in (rng.integers(0, len(keys), len(keys)) for _ in range(B))]
    return float(s.sum() / n.sum()), float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))
