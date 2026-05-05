import numpy as np
from itertools import combinations, product
import random
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.util.misc import get_duplicates
from scipy.spatial.distance import cdist

class ExhaustiveSubsetGeneration:
    def __init__(self, distance_threshold=None):
        # We store hashes of IDs to save memory
        self.memory = set()
        self.distance_threshold = distance_threshold
    
    def _get_pair_id(self, c1, c2):
        # Create a unique, order-independent ID for the pair
        id1, id2 = id(c1), id(c2)
        return (id1, id2) if id1 < id2 else (id2, id1)

    def filter_and_memory(self, couples):
        valid_couples = []
        
        for c1, c2 in couples:
            # 1. Memory Check (using object IDs)
            pair_id = self._get_pair_id(c1, c2)
            if pair_id in self.memory:
                continue
            
            # 2. Distance Check (Fixed number of items using XOR)
            if self.distance_threshold is not None:
                # Count how many items differ
                diff = np.count_nonzero(c1.X != c2.X)
                if diff <= self.distance_threshold:
                    continue
            
            # If it passes both, add to memory and result
            self.memory.add(pair_id)
            valid_couples.append((c1, c2))
            
        return valid_couples
                
    def generate(self, reference_set):
        new_solutions, old_solutions = reference_set
        old_solutions = old_solutions if old_solutions is not None else []

        if not old_solutions:
            couples = list(combinations(new_solutions, 2))
        else:
            # Standard Scatter Search rule: at least one solution must be "new"
            couples = list(combinations(new_solutions, 2)) + list(product(new_solutions, old_solutions))
        
        return self.filter_and_memory(couples)


class BinaryTournamentSubsetGeneration:

    def __init__(self, distance_threshold=None):
        self.rng = np.random.default_rng()
        self.memory = set()
        self.distance_threshold = distance_threshold

    def set_seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def _pair_key(self, a, b):
        # Optimized ID-based hash
        ia, ib = id(a), id(b)
        return (ia, ib) if ia < ib else (ib, ia)

    def generate(self, reference_set):
        new_solutions, old_solutions = reference_set
        all_solutions = list(new_solutions)
        if old_solutions is not None:
            all_solutions.extend(old_solutions)

        n = len(all_solutions)
        if n < 2: return []

        # Tournament selection
        idx = np.arange(n)
        p1 = self.rng.permutation(idx)
        p2 = self.rng.permutation(idx)
        combined = np.concatenate([p1, p2]).reshape(-1, 2)

        F = np.array([sol.F[0] for sol in all_solutions])
        winners = np.where(F[combined[:, 0]] < F[combined[:, 1]], combined[:, 0], combined[:, 1])

        if len(winners) % 2 == 1:
            winners = winners[:-1]
        winner_pairs_idx = winners.reshape(-1, 2)

        # Vectorized Distance Filter
        X = np.array([all_solutions[i].X for i in winners])
        X1, X2 = X[0::2], X[1::2]

        if self.distance_threshold is not None:
            # Fixed number of items different (Hamming Distance)
            dists = np.sum(X1 != X2, axis=1)
            keep = dists > self.distance_threshold
            winner_pairs_idx = winner_pairs_idx[keep]

        # Memory Filter
        pairs = []
        for i, j in winner_pairs_idx:
            a, b = all_solutions[i], all_solutions[j]
            key = self._pair_key(a, b)

            if key not in self.memory:
                self.memory.add(key)
                pairs.append((a, b))

        return pairs


class TournamentSubsetGeneration2:
    def __init__(self, tournament_k=2, n_pairs=None, seed=None):
        self.tournament_k = tournament_k
        self.n_pairs = n_pairs
        self.rng = np.random.default_rng(seed)

    def set_seed(self, seed):
        self.rng = np.random.default_rng(seed)

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
            p1 = min(self.rng.choice(all_solutions, min(self.tournament_k, len(all_solutions)), replace=False), key=lambda x: x.F[0])
            
            # 2. Definir de dónde podemos sacar a p2
            if id(p1) in old_ids:
                # Si p1 es viejo, p2 DEBE ser nuevo obligatoriamente
                pool_p2 = list(new_solutions)
            else:
                # Si p1 es nuevo, p2 puede venir de cualquier lado (excepto p1)
                pool_p2 = [s for s in all_solutions if s is not p1]
            
            # 3. Seleccionar p2 del pool correspondiente
            p2 = min(self.rng.choice(pool_p2, min(self.tournament_k, len(pool_p2)), replace=False), key=lambda x: x.F[0])

            
            pairs.append((p1, p2))
            
        return pairs
        
class PymooCrossoverCombination:

    def __init__(self, crossover_operator, problem=None):
        self.crossover = crossover_operator
        self.problem = problem
        self.rng = np.random.default_rng()

    def set_seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def set_problem(self, problem):
        self.problem = problem

    def problem_is_set(self):
        return self.problem is not None

    def combine(self, pairs, threshold=0):

        if not self.problem_is_set():
            raise ValueError("Problem is not set")

        if not pairs:
            return []

        matings = [[p1, p2] for p1, p2 in pairs]

        offspring_pop = self.crossover.do(
            self.problem,
            matings,
            random_state=self.rng
        )

        
        X = offspring_pop.get("X")
        _, idx = np.unique(X, axis=0, return_index=True)
        offspring_pop = offspring_pop[np.sort(idx)]

        return offspring_pop

class PathRelinking_RCL:
    def __init__(self, alpha=0.5, max_candidates=5, max_steps=1):
        self.problem = None
        self.evaluator = None
        self.alpha = alpha
        self.max_candidates = max_candidates
        self.max_steps = max_steps
        self.rng = np.random.default_rng()

    def set_seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def set_problem(self, problem):
        self.problem = problem

    def set_evaluator(self, evaluator):
        self.evaluator = evaluator

    def _get_evaluator(self):
        from pymoo.core.evaluator import Evaluator
        return self.evaluator if self.evaluator is not None else Evaluator()

    def combine(self, pairs):
        """
        Punto de entrada para Scatter Search.
        'pairs' es una lista de tuplas (Indiv1, Indiv2).
        """
        my_pairs = pairs

        offspring = []
        for p1, p2 in my_pairs:
            # Relinking en ambas direcciones (bidireccional)
            offspring.append(self._relink(p1, p2))
            offspring.append(self._relink(p2, p1))

        return offspring

    def _relink(self, sol_start, sol_target):
        ev = self._get_evaluator()
        max_candidates = self.max_candidates
        x_curr = np.copy(sol_start.X)
        target = sol_target.X
        
        best_x = np.copy(x_curr)
        best_f = sol_start.F[0]
    
        # Índices donde las soluciones difieren
        diff_indices = np.where(x_curr != target)[0]
        steps = 0
    
        # Mientras existan diferencias y no hayamos llegado al objetivo
        while len(diff_indices) > 0 or steps < self.max_steps:
            # --- REDUCCIÓN DE EVALUACIONES: Muestreo Agresivo ---
            steps += 1
            # No evaluamos todos los movimientos, solo una pequeña muestra aleatoria
            n_to_sample = min(len(diff_indices), max_candidates)
            # Seleccionamos posiciones aleatorias de la lista de diferencias
            sample_indices_in_diff = self.rng.choice(len(diff_indices), n_to_sample, replace=False)
            current_sample = diff_indices[sample_indices_in_diff]
        
            # Crear candidatos para evaluar
            batch_X = np.tile(x_curr, (len(current_sample), 1))
            for i, idx in enumerate(current_sample):
                batch_X[i, idx] = target[idx]
        
            # Evaluación por lotes
            pop_cand = Population.new("X", batch_X)
            ev.eval(self.problem, pop_cand)
        
            # Extraer f_vals asegurando que sea un vector (N,)
            f_vals = pop_cand.get("F")
            if f_vals is None or len(f_vals) == 0: break
            f_vals = f_vals[:, 0] 
        
            # --- Lógica RCL ---
            f_min, f_max = np.min(f_vals), np.max(f_vals)
            if f_max == f_min:
                idx_in_f_vals = int(self.rng.integers(0, len(f_vals)))
            else:
                threshold = f_min + self.alpha * (f_max - f_min)
                rcl_indices = np.where(f_vals <= threshold)[0]
                idx_in_f_vals = int(self.rng.choice(rcl_indices))
        
            # --- ACTUALIZACIÓN DE ESTADO (Aquí estaba el error) ---
            # 1. Obtenemos el índice real del vector X
            global_idx = current_sample[idx_in_f_vals]
        
            # 2. Aplicamos el movimiento
            x_curr[global_idx] = target[global_idx]
        
            # 3. Guardamos si es mejor que lo visto en este camino
            if f_vals[idx_in_f_vals] < best_f:
                best_f = f_vals[idx_in_f_vals]
                best_x = np.copy(x_curr)
        
            # 4. ACTUALIZACIÓN CRÍTICA: Recalcular diff_indices
            # Es más seguro recalcular o filtrar el índice usado
            diff_indices = np.delete(diff_indices, sample_indices_in_diff[idx_in_f_vals])

        return Individual(X=best_x, F=np.array([best_f]))