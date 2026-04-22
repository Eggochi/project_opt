from pymoo.core.problem import Problem
import numpy as np
#nk landscape

class NKLandscape(Problem):
    def __init__(self, N, K, seed=42):
        super().__init__(n_var=N, n_obj=1, n_constr=0, xl=0, xu=1)
        self.N = N
        self.K = K
        rng = np.random.default_rng(seed)
        # Generate random neighbors for each component
        # Each component i has a table of fitness values based on K neighbors
        # For simplicity, we use random neighbors for each component
        self.neighbors = np.zeros((N, K), dtype=int)
        for i in range(N):
            potential_neighbors = [j for j in range(N) if j != i]
            # Fix: Use rng.choice instead of rng.sample
            self.neighbors[i] = rng.choice(potential_neighbors, K, replace=False)

        # Generate random fitness table [0, 1] for each component
        # Table size is 2^(K+1)
        self.fitness_tables = rng.random((N, 2**(K+1)))

        #Forced max sequence
        max_sequence= np.zeros(N, dtype=int)
        for i in range(N):
          if i%2==0:
            max_sequence[i]=1

        all_indices = np.hstack([np.arange(N)[:, None], self.neighbors])
        max_bits = max_sequence[all_indices] # Shape (N, K+1)
        powers = 2 ** np.arange(self.K, -1, -1)
        table_idx = np.sum(max_bits * powers, axis=1)

        # Vectorized update
        self.fitness_tables[np.arange(N), table_idx] = 0.9999


    def _evaluate(self, X, out, *args, **kwargs):
       # X shape: (n_individuals, N)
      X_int = X.astype(int)
      n_individuals, N = X_int.shape

      # 1. Prepare indices for all neighbors
      # neighbors_idx shape: (N, K+1) where K is the number of neighbors
      # This assumes self.neighbors is a 2D array or list of lists of equal length
      indices = np.hstack([np.arange(N)[:, None], self.neighbors])

      # 2. Gather bits for all individuals at once
      # Resulting shape: (n_individuals, N, K+1)
      relevant_bits = X_int[:, indices]

      # 3. Convert bits to table indices using powers of 2
      # Create an array of [2^K, 2^{K-1}, ..., 2^0]
      K_plus_1 = relevant_bits.shape[2]
      powers = 2 ** np.arange(K_plus_1 - 1, -1, -1)

      # table_indices shape: (n_individuals, N)
      table_indices = np.sum(relevant_bits * powers, axis=2)

      # 4. Advanced Indexing to pull from fitness_tables
      # fitness_tables shape: (N, 2^{K+1})
      # We want to pick fitness_tables[i, table_indices[k, i]]
      row_idx = np.arange(N)
      # Using broadcasting to pull values for all individuals
      all_fitness_contributions = self.fitness_tables[row_idx, table_indices]

      # 5. Final calculation (averaging and negation)
      fitness_values = -np.mean(all_fitness_contributions, axis=1)

      out['F'] = fitness_values


class QUBO(Problem):
  def __init__(self, Q):
    super().__init__(n_var=Q.shape[0], n_obj=1, n_constr=0, xl=0, xu=1)
    self.Q = Q
    self.counter=0

  def _evaluate(self, X, out, *args, **kwargs):
    X_int=X.astype(int)
    out['F'] = np.sum(X_int @ self.Q * X_int, axis=1)
    self.counter+=1