import numpy as np
from scipy.stats import binom


def binom_euro_call(r, u, d, S0, K, T, n, h):
    q = (np.exp(r * T / n) - d) / (u - d)

    expected = binom.expect(h, (n, q))

    return expected


def binom_usa_call(r, u, d, S0, K, T, n, h):
    pass


if __name__ == "__main__":
    print(binom_euro_call(1,1,1,1,1,1,10, lambda x: x))
