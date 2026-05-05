import numpy as np
import random
from pymoo.core.population import Population

class FrequencyBinaryDiversification:

    def __init__(self):
        self.rng = np.random.default_rng()

    def set_seed(self, seed):
        self.rng = np.random.default_rng(seed)

    
    def set_problem(self, problem):
        self.problem = problem

    def do(self, problem, n_samples, seed_population=None):
        if problem is not None:
            self.problem = problem

        if self.problem is None:
            raise ValueError("Problem not set. Call set_problem() first.")

        n_vars = self.problem.n_var

        if seed_population is None or len(seed_population) == 0:
            true_counts = np.zeros(n_vars, dtype=float)
            offset = 0
        else:
            X_seed = np.asarray([ind.X for ind in seed_population], dtype=bool)
            true_counts = X_seed.sum(axis=0).astype(float)
            offset = len(X_seed)

        X = np.empty((n_samples, n_vars), dtype=bool)

        for i in range(n_samples):
            total_so_far = i + offset
            false_counts = total_so_far - true_counts

            prob_true = (false_counts + 1.0) / (total_so_far + 2.0)

            row_bits = self.rng.random(n_vars) < prob_true


            X[i] = row_bits
            true_counts += row_bits.astype(float)

        pop = Population.new("X", X)
        return pop