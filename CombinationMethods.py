import numpy as np
from itertools import combinations, product
import random
from pymoo.core.individual import Individual
from pymoo.core.population import Population

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

import numpy as np

class BinaryTournamentSubsetGeneration:
    def __init__(self):
        pass
        
    def generate(self, reference_set):
        new_solutions, old_solutions = reference_set
        
        # 1. Unificar todas las soluciones en una lista
        all_solutions = list(new_solutions)
        if old_solutions is not None:
            all_solutions.extend(old_solutions)
        
        n = len(all_solutions)
        # Necesitamos al menos 2 soluciones para formar 1 pareja
        if n < 2: return [] 

        # 2. Generar y barajar índices
        indices = np.arange(n)
        p1 = indices.copy()
        p2 = indices.copy()
        np.random.shuffle(p1)
        np.random.shuffle(p2)
        
        # Concatenamos para procesar todo en un solo bucle
        combined_idx = np.concatenate([p1, p2])
        # n es par, así que combined_idx (2n) siempre es divisible entre 2
        pairs_idx = np.split(combined_idx, len(combined_idx) // 2)

        # 3. Torneo: Guardamos solo los índices de los ganadores
        winners = []
        for i, j in pairs_idx:
            if all_solutions[i].F[0] < all_solutions[j].F[0]:
                winners.append(i)
            else:
                winners.append(j)
        
        # 4. Formar pares finales con los objetos reales
        pairs = []
        # Iteramos de 2 en 2 para emparejar ganadores
        for k in range(0, len(winners) - 1, 2):
            if k + 1 < len(winners):
                idx1 = winners[k]
                idx2 = winners[k+1]
                pairs.append((all_solutions[idx1], all_solutions[idx2]))

            
        return pairs


class PymooCrossoverCombination:
    def __init__(self, crossover_operator, problem=None):
        self.crossover = crossover_operator
        self.problem = problem
        self.comm = None

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
    def __init__(self, alpha=0.5, max_candidates=5, max_steps=1):
        self.problem = None
        self.evaluator = None
        self.comm = None
        
        self.alpha = alpha
        self.max_candidates = max_candidates
        self.max_steps = max_steps
        

    def set_problem(self, problem):
        self.problem = problem

    def set_evaluator(self, evaluator):
        self.evaluator = evaluator

    def set_comm(self, comm):
        self.comm = comm

    def _get_evaluator(self):
        from pymoo.core.evaluator import Evaluator
        return self.evaluator if self.evaluator is not None else Evaluator()

    def combine(self, pairs):
        """
        Punto de entrada para Scatter Search.
        'pairs' es una lista de tuplas (Indiv1, Indiv2).
        """
        # Distribución de trabajo para MPI
        if self.comm is not None:
            size = self.comm.Get_size()
            rank = self.comm.Get_rank()
            my_pairs = np.array_split(pairs, size)[rank]
        else:
            my_pairs = pairs

        offspring = []
        for p1, p2 in my_pairs:
            # Relinking en ambas direcciones (bidireccional)
            offspring.append(self._relink(p1, p2))
            offspring.append(self._relink(p2, p1))

        # Sincronización MPI
        if self.comm is not None:
            gathered = self.comm.allgather(offspring)
            return [s for sublist in gathered for s in sublist]
        
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
            sample_indices_in_diff = np.random.choice(len(diff_indices), n_to_sample, replace=False)
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
            f_vals = f_vals[:, 0] # Tomamos solo el primer objetivo si es QUBO
        
            # --- Lógica RCL ---
            f_min, f_max = np.min(f_vals), np.max(f_vals)
            if f_max == f_min:
                idx_in_f_vals = random.randrange(len(f_vals))
            else:
                threshold = f_min + self.alpha * (f_max - f_min)
                rcl_indices = np.where(f_vals <= threshold)[0]
                idx_in_f_vals = random.choice(rcl_indices)
        
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