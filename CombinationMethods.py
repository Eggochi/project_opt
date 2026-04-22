import numpy as np
from itertools import combinations, product
import random
from pymoo.core.individual import Individual

class ExhaustiveSubsetGeneration:
    def __init__(self):
        pass
    
    def generate(self, reference_set):
        new_solutions, old_solutions = reference_set
        pairs = list(combinations(new_solutions, 2))
        if old_solutions is not None:
            pairs.extend(product(new_solutions, old_solutions))
        return pairs

class TournamentSubsetGeneration:
    def __init__(self, tournament_k=2, n_pairs=None):
        self.tournament_k = tournament_k
        self.n_pairs = n_pairs
        
    def generate(self, reference_set):
        new_solutions, old_solutions = reference_set
        
        all_solutions = list(new_solutions)
        if old_solutions is not None:
            all_solutions.extend(old_solutions)
            
        n_pairs = self.n_pairs
        if n_pairs is None:
            num_new = len(new_solutions)
            num_old = len(old_solutions) if old_solutions is not None else 0
            n_pairs = (num_new * (num_new - 1)) // 2 + num_new * num_old

        pairs = []
        for _ in range(n_pairs):
            p1 = min(random.sample(all_solutions, min(self.tournament_k, len(all_solutions))), key=lambda x: x.F[0])
            p2 = min(random.sample(all_solutions, min(self.tournament_k, len(all_solutions))), key=lambda x: x.F[0])
            if p1 is p2 and len(all_solutions) > 1:
                pool_without_p1 = [s for s in all_solutions if s is not p1]
                p2 = min(random.sample(pool_without_p1, min(self.tournament_k, len(pool_without_p1))), key=lambda x: x.F[0])
            pairs.append((p1, p2))
            
        return pairs

import random

class TournamentSubsetGeneration2:
    def __init__(self, tournament_k=2, n_pairs=None):
        self.tournament_k = tournament_k
        self.n_pairs = n_pairs
        
    def generate(self, reference_set):
        new_solutions, old_solutions = reference_set
        
        # Manejo seguro en caso de que old_solutions sea None
        old_solutions = old_solutions if old_solutions is not None else []
        all_solutions = list(new_solutions) + list(old_solutions)
        
        # Guardamos los IDs de las soluciones viejas en un Set para búsqueda en O(1)
        old_ids = {id(s) for s in old_solutions}
        
        n_pairs = self.n_pairs
        if n_pairs is None:
            num_new = len(new_solutions)
            num_old = len(old_solutions)
            # La fórmula original es correcta: combinaciones (new, new) + (new, old)
            n_pairs = (num_new * (num_new - 1)) // 2 + num_new * num_old

        pairs = []
        
        # Si no hay soluciones nuevas y se exigen pares, devolvemos vacío 
        # para no violar la regla de "cero pares old-old" y evitar crasheos.
        if not new_solutions and n_pairs > 0:
            return pairs

        for _ in range(n_pairs):
            # 1. Seleccionar p1 haciendo torneo sobre TODAS las soluciones
            p1 = min(random.sample(all_solutions, min(self.tournament_k, len(all_solutions))), key=lambda x: x.F[0])
            
            # 2. Definir de dónde podemos sacar a p2
            if id(p1) in old_ids:
                # Si p1 es viejo, p2 DEBE ser nuevo obligatoriamente
                pool_p2 = list(new_solutions)
            else:
                # Si p1 es nuevo, p2 puede venir de cualquier lado (excepto p1)
                pool_p2 = [s for s in all_solutions if s is not p1]
            
            # 3. Seleccionar p2 del pool correspondiente
            p2 = min(random.sample(pool_p2, min(self.tournament_k, len(pool_p2))), key=lambda x: x.F[0])
            
            pairs.append((p1, p2))
            
        return pairs

class PymooCrossoverCombination:
    def __init__(self, crossover_operator, problem=None):
        self.crossover = crossover_operator
        self.problem = problem

    def set_problem(self, problem):
        self.problem = problem

    def set_comm(self, comm):
        self.comm = comm

    def problem_is_set(self):
        return self.problem is not None
        
    def combine(self, pairs, threshold=0):
        if not self.problem_is_set():
            raise ValueError("Problem is not set")
            
        if not pairs:
            return []

        # --- MPI Parallel Work Distribution ---
        if hasattr(self, 'comm') and self.comm is not None:
            size = self.comm.Get_size()
            rank = self.comm.Get_rank()
            my_pairs = list(np.array_split(pairs, size)[rank])
        else:
            my_pairs = pairs
        
        # Filtrar pares por distancia de Hamming (or other metrics)
        matings = []
        for p1, p2 in my_pairs:
            distance = np.sum(p1.X != p2.X)
            if distance >= threshold * len(p1.X):
                matings.append([p1, p2])
    
        if not matings:
            offspring = []
        else:
            offspring_pop = self.crossover.do(self.problem, matings)
            offspring = list(offspring_pop)

        # --- MPI Gather ---
        if hasattr(self, 'comm') and self.comm is not None:
            gathered = self.comm.allgather(offspring)
            return [s for sublist in gathered for s in sublist]
        else:
            return offspring

class PathRelinking_RCL:
    def __init__(self, problem, alpha=0.5, evaluator=None):
        self.problem = problem
        self.alpha = alpha
        from pymoo.core.evaluator import Evaluator
        self.evaluator = evaluator if evaluator is not None else Evaluator()

    def set_comm(self, comm):
        self.comm = comm
        
    def combine(self, pairs):
        # --- MPI Parallel Work Distribution ---
        if hasattr(self, 'comm') and self.comm is not None:
            size = self.comm.Get_size()
            rank = self.comm.Get_rank()
            my_pairs = list(np.array_split(pairs, size)[rank])
        else:
            my_pairs = pairs

        # Using a list comprehension here is fine, but we process pairs
        offspring = [res for p in my_pairs for res in (self._relink(p[0], p[1]), self._relink(p[1], p[0]))]

        # --- MPI Gather ---
        if hasattr(self, 'comm') and self.comm is not None:
            gathered = self.comm.allgather(offspring)
            return [s for sublist in gathered for s in sublist]
        else:
            return offspring

    def _relink(self, sol_start, sol_target):
        # 1. Setup initial state
        x = np.array(sol_start.X).copy()
        target = np.array(sol_target.X).copy()
        
        x_best = x.copy()
        f_best = sol_start.F[0]
        
        # 2. Track indices that need to change
        # This is faster than re-scanning the whole array inside the loop
        remaining_diffs = np.where(x != target)[0]
        
        while len(remaining_diffs) > 0:
            n_candidates = len(remaining_diffs)
            
            # 3. Vectorized Candidate Generation
            # Pre-allocate the matrix for speed
            batch_X = np.empty((n_candidates, x.shape[0]), dtype=x.dtype)
            batch_X[:] = x
            
            # Apply one unique move per row
            for i, idx in enumerate(remaining_diffs):
                batch_X[i, idx] = target[idx]
            
            # 4. Batch Evaluation
            pop = Population.new("X", batch_X)
            self.evaluator.eval(self.problem, pop)
            f_vals = pop.get("F").flatten()
            
            # 5. RCL Logic
            f_min, f_max = f_vals.min(), f_vals.max()
            
            # If all improvements are equal, threshold logic can be simplified
            if f_max == f_min:
                chosen_local_idx = random.randrange(n_candidates)
            else:
                threshold = f_min + self.alpha * (f_max - f_min)
                rcl_indices = np.where(f_vals <= threshold)[0]
                chosen_local_idx = random.choice(rcl_indices)
            
            # 6. State Update
            # Get the global index of the move we just made
            moved_idx = remaining_diffs[chosen_local_idx]
            
            x[moved_idx] = target[moved_idx]
            f_next = f_vals[chosen_local_idx]
            
            # Update best-so-far
            if f_next < f_best:
                x_best = x.copy()
                f_best = f_next
            
            # 7. Remove the used index from our "to-do" list
            # Much faster than np.where() on the whole array
            remaining_diffs = np.delete(remaining_diffs, chosen_local_idx)

        # Return result as a pymoo Individual
        res = Individual(X=x_best)
        res.F = np.array([f_best])
        return res