from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
import numpy as np
#nk landscape

class RepairKnapsack(Repair):
    def _do(self, problem, Z, **kwargs):
      
      #max capacity
      Q=problem.C

      ratio = problem.P / problem.W
      
      # weights if each individual
      weights = (Z*problem.W).sum(axis=1)
      
      #repair each individual
      for i in range(len(Z)):
        
        # packing plan for i
        z=Z[i]
        
          
        # while the maximum capacity violation exist
        while weights[i]>Q:
          # randomly selectn an item currently picked
            selected = np.flatnonzero(z)
            # Remove selected item with worst profit/weight ratio
            item_to_remove = selected[np.argmin(ratio[selected])]
          
            z[item_to_remove]=False
            weights[i]-=problem.W[item_to_remove]
          
      return Z

class RepairKnapsack2(Repair):
    def __init__(self, alpha=2):
        super().__init__()
        self.alpha = alpha
    def _do(self, problem, Z, **kwargs):
        # Max capacity
        Q = problem.C
        # Profit/Weight ratio
        ratio = problem.P / problem.W
        
        # Calculate current weights
        # Ensure Z is boolean for easier indexing
        Z = Z.astype(bool)
        weights = (Z * problem.W).sum(axis=1)
        
        # We need a random generator
        rng = np.random.default_rng()

        for i in range(len(Z)):
            z = Z[i]
            
            while weights[i] > Q:
                selected = np.flatnonzero(z)
                
                # 1. Get the ratios of currently selected items
                selected_ratios = ratio[selected]
                
                # 2. Convert ratios to "removal weights" 
                # Lower ratio = Higher chance to be removed.
                # Adding a small epsilon (1e-10) avoids division by zero
                removal_weights = 1.0 / (selected_ratios**self.alpha + 1e-10)
                
                # 3. Normalize weights to create probabilities
                probs = removal_weights / removal_weights.sum()
                
                # 4. Probabilistically select an item to remove
                item_to_remove = rng.choice(selected, p=probs)
                
                # Apply removal
                z[item_to_remove] = False
                weights[i] -= problem.W[item_to_remove]
                
        return Z

class RandomRepairKnapsack(Repair):
    def _do(self, problem, Z, **kwargs):
      
      #max capacity
      Q=problem.C

      ratio = problem.P / problem.W
      
      # weights if each individual
      weights = (Z*problem.W).sum(axis=1)
      
      #repair each individual
      for i in range(len(Z)):
        
        # packing plan for i
        z=Z[i]
        
          
        # while the maximum capacity violation exist
        while weights[i]>Q:
          # randomly selectn an item currently picked
            item_to_remove = np.random.choice(np.where(z)[0])

            z[item_to_remove]=False
            weights[i]-=problem.W[item_to_remove]
          
      return Z
      

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

class KnapsackProblem(Problem):
    def __init__(self, filename, obj_idx=0, penalty=1e6):
        self.filename = filename
        self.obj_idx = obj_idx
        self.penalty = penalty
        
        # Reader logic
        weights = []
        profits = []
        n_items = 0
        n_objs = 0
        
        with open(filename, 'r') as f:
            mode = None
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if parts[0] == 'p':
                    if parts[1] == 'KNAPSACK':
                        n_items = int(parts[2])
                        n_objs = int(parts[3])
                    elif parts[1] == 'weights':
                        mode = 'weights'
                    elif parts[1] == 'profits':
                        mode = 'profits'
                    continue
                
                vals = [float(x) for x in parts]
                if mode == 'weights':
                    weights.append(vals)
                elif mode == 'profits':
                    profits.append(vals)
        
        self.weights = np.array(weights)
        self.profits = np.array(profits)
        
        # Select objective
        self.w = self.weights[:, obj_idx]
        self.p = self.profits[:, obj_idx]
        self.C = 0.5 * np.sum(self.w)
        
        super().__init__(n_var=n_items, n_obj=1, n_constr=1, xl=0, xu=1)

    def _evaluate(self, X, out, *args, **kwargs):
        X_int = X.astype(int)
        
        # Total profit (to maximize, so negative for pymoo minimize)
        profit = np.sum(X_int * self.p, axis=1)
        # Total weight
        weight = np.sum(X_int * self.w, axis=1)
        
        # Constraint violation
        cv = np.maximum(0, weight - self.C)
        
        # F with penalty for the existing LocalSearch to work
        if self.penalty:
            out["F"] = -profit + self.penalty * cv
        else:
            out["F"] = -profit
            
        out["G"] = cv

