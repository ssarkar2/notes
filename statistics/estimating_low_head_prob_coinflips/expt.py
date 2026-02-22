def sum_terms(p=0.1, n_terms=100):
    total = 0.0
    for k in range(1, n_terms + 1):
        term = (2 / (k + 2)) * ((1 - p) ** (k - 1)) * p
        total += term
    return total


result = sum_terms()
print(f"Sum over 100 terms for p=0.1: {result-0.1}")

def get_bias_fn_given_alpha_beta(alpha, beta):
    def bias_fn(h):
        bias = (alpha + 1) * h * sum([(1 - h) ** (n-1) / (n + alpha + beta) for n in range(1, 1000)]) - h
        return bias
    return bias_fn

bias_fn = get_bias_fn_given_alpha_beta(1, 1)
print(bias_fn(0.1))

import numpy as np
def bayesian_uniform_bias(h):
    k_max = 1000
    k = np.arange(1, k_max + 1)
    terms = (2 / (k + 2)) * h * (1 - h) ** (k - 1)
    return np.sum(terms) - h


print(bayesian_uniform_bias(0.1))