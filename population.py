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


class FrequencyBinaryDiversification2:
    """
    Versión optimizada y vectorizada de la diversificación por frecuencia.
    Elimina el bucle interno de variables para ganar velocidad.
    """
    
    def do(self, problem, n_samples, seed_population=None):
        n_vars = problem.n_var
        
        # 1. Inicializar conteos (soporta semillas previas)
        if seed_population is None:
            true_counts = np.zeros(n_vars)
            offset = 0
        else:
            # Convertimos la población semilla a matriz y sumamos columnas
            X_seed = np.array([ind.X for ind in seed_population])
            true_counts = X_seed.sum(axis=0).astype(float)
            offset = len(seed_population)
        
        # 2. Pre-asignar la matriz de resultados
        X = np.empty((n_samples, n_vars), dtype=bool)
        
        # 3. Generación por filas (el bucle de muestras es necesario por la dependencia)
        # Pero las operaciones dentro son vectorizadas (NumPy)
        for i in range(n_samples):
            total_so_far = i + offset
            false_counts = total_so_far - true_counts
            
            # Simplificación matemática de: w_true / (w_true + w_false)
            # Prob = (1/(T+1)) / (1/(T+1) + 1/(F+1)) -> (F+1) / (T+F+2)
            prob_true = (false_counts + 1.0) / (total_so_far + 2.0)
            
            # Generar toda la fila de una vez
            row_bits = np.random.random(n_vars) < prob_true
            X[i, :] = row_bits
            
            # Actualizar memoria de frecuencia (vectorizado)
            true_counts += row_bits
            
        return Population.new("X", X)