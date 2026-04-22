import numpy as np
import random
from pymoo.core.population import Population

class FrequencyBinaryDiversification:
    """
    Diversification method for Scatter Search that ensures 
    a balanced distribution of True/False values across 
    the initial solution pool using frequency memory.
    """
    
    def do(self, problem, n_samples):
        # 1. Setup metadata from the pymoo problem
        n_vars = problem.n_var
        
        # 2. Track how many times each bit position has been set to True
        true_counts = np.zeros(n_vars)
        
        # 3. Pre-allocate the boolean matrix (X)
        X = np.empty((n_samples, n_vars), dtype=bool)
        
        # 4. Generate solutions one by one
        for i in range(n_samples):
            # 'i' represents the number of solutions already generated
            solutions_so_far = i 
            
            for v in range(n_vars):
                count_true = true_counts[v]
                count_false = solutions_so_far - count_true
                
                # Inverse weights: prefer the state used less frequently
                w_true = 1.0 / (count_true + 1)
                w_false = 1.0 / (count_false + 1)
                
                # Normalize probability
                prob_true = w_true / (w_true + w_false)
                
                # Assign value and update memory
                if random.random() < prob_true:
                    X[i, v] = True
                    true_counts[v] += 1
                else:
                    X[i, v] = False
        
        # 5. Return as a pymoo Population object
        return Population.new("X", X)