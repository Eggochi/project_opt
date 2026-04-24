import numpy as np
from pymoo.core.population import Population
from pymoo.operators.mutation.bitflip import BitflipMutation

class LocalSearchImprovement:
    def __init__(self, problem=None, max_steps=None, neighbor_sample=None, use_cache=False):
        self.problem = problem
        self.max_steps = max_steps
        self.neighbor_sample = neighbor_sample
        self.use_cache = use_cache
        self.cache = {} if use_cache else None
        self.evaluator = None
        self.comm = None
        self.rng = np.random.default_rng()

    def set_seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def set_comm(self, comm):
        self.comm = comm

    def set_problem(self, problem):
        self.problem = problem

    def set_evaluator(self, evaluator):
        self.evaluator = evaluator



    def _get_evaluator(self):
        from pymoo.core.evaluator import Evaluator as _Ev
        return self.evaluator if self.evaluator is not None else _Ev()

    def problem_is_set(self):
        return self.problem is not None

    def _evaluate_batch(self, X_arr):
        if not self.problem_is_set():
            raise ValueError("Problem is not set")
        ev = self._get_evaluator()

        if not self.use_cache:
            pop = Population.new("X", X_arr)
            ev.eval(self.problem, pop)
            return pop.get("F").flatten()

        n_samples = len(X_arr)
        F = np.zeros(n_samples)
        keys = [x.tobytes() for x in X_arr]

        to_eval_idx = []
        to_eval_X = []

        for i in range(n_samples):
            k = keys[i]
            if k in self.cache:
                F[i] = self.cache[k]
            else:
                to_eval_idx.append(i)
                to_eval_X.append(X_arr[i])

        if to_eval_X:
            to_eval_X = np.array(to_eval_X)
            sub_pop = Population.new("X", to_eval_X)
            ev.eval(self.problem, sub_pop)
            f_new = sub_pop.get("F").flatten()

            for j, val in enumerate(f_new):
                orig_idx = to_eval_idx[j]
                self.cache[keys[orig_idx]] = val
                F[orig_idx] = val

        return F

    def improve_pool(self, solutions):
        if not self.problem_is_set():
            raise ValueError("Problem is not set")

        ev = self._get_evaluator()
        if not isinstance(solutions, Population):
            if not solutions:
                return []
            solutions = Population.create(*solutions)
        
        if len(solutions) == 0:
            return solutions

        # --- MPI Parallel Work Distribution ---
        if hasattr(self, 'comm') and self.comm is not None:
            size = self.comm.Get_size()
            rank = self.comm.Get_rank()
            my_solutions = list(np.array_split(list(solutions), size)[rank])
        else:
            my_solutions = list(solutions)

        for sol in my_solutions:
            x = sol.X.copy()
            n = len(x)

            if not hasattr(sol, 'F') or sol.F is None or (isinstance(sol.F, np.ndarray) and sol.F.size == 0):
                init_pop = Population.new("X", np.array([x]))
                ev.eval(self.problem, init_pop)
                f_current = float(init_pop.get("F").flat[0])
            else:
                f_current = float(np.asarray(sol.F).flat[0])

            improved = True
            steps = 0
            k = min(self.neighbor_sample, n) if self.neighbor_sample is not None else n
            neighbors_buffer = np.empty((k, n), dtype=x.dtype)
            row_idx = np.arange(k)

            while improved and (self.max_steps is None or steps < self.max_steps):
                improved = False
                if self.neighbor_sample is not None:
                    indices = self.rng.choice(n, k, replace=False)
                else:
                    indices = self.rng.permutation(n)

                neighbors_buffer[:] = x
                if x.dtype == bool:
                    neighbors_buffer[row_idx, indices] = ~x[indices]
                else:
                    neighbors_buffer[row_idx, indices] = 1 - x[indices]

                f_vals = self._evaluate_batch(neighbors_buffer)
                check = f_vals < f_current
                if np.any(check):
                    first_better = np.where(check)[0][0]
                    x = neighbors_buffer[first_better].copy()
                    f_current = f_vals[first_better]
                    improved = True
                    steps += 1
            
            sol.X = x
            sol.F = np.array([f_current])

        # --- MPI Gather ---
        if hasattr(self, 'comm') and self.comm is not None:
            gathered = self.comm.allgather(my_solutions)
            all_sols = [s for sublist in gathered for s in sublist]
            return Population.create(*all_sols)
        else:
            return Population.create(*my_solutions)


class LocalSearch_BitFlipMutation:
    def __init__(self, prob_var=0.05, problem=None, max_steps=1, n_neighbors=5):
        self.problem = problem
        self.max_steps = max_steps
        self.n_neighbors = n_neighbors
        self.prob_var = prob_var
        self.evaluator = None
        self.comm = None
        self.rng = np.random.default_rng()

    def set_seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def set_problem(self, problem):
        self.problem = problem

    def set_evaluator(self, evaluator):
        self.evaluator = evaluator

    def set_comm(self, comm):
        self.comm = comm

    def _get_evaluator(self):
        from pymoo.core.evaluator import Evaluator as _Ev
        return self.evaluator if self.evaluator is not None else _Ev()

    def improve_pool(self, solutions):
        if self.problem is None:
            raise ValueError("Problem is not set")

        ev = self._get_evaluator()
        if not isinstance(solutions, Population):
            if not solutions:
                return []
            solutions = Population.create(*solutions)
        
        if len(solutions) == 0:
            return solutions

        # --- MPI Parallel Work Distribution ---
        if hasattr(self, 'comm') and self.comm is not None:
            size = self.comm.Get_size()
            rank = self.comm.Get_rank()
            my_sols_list = list(np.array_split(list(solutions), size)[rank])
            my_solutions = Population.create(*my_sols_list) if len(my_sols_list) > 0 else Population()
        else:
            my_solutions = solutions

        if len(my_solutions) > 0:
            ev.eval(self.problem, my_solutions)
            N = len(my_solutions)

            for _ in range(self.max_steps):
                X = my_solutions.get("X")  # shape (N, n_vars)
                X_rep = np.repeat(X, self.n_neighbors, axis=0)  # (N*n_neighbors, n_vars)

                # Manual bit-flip using the isolated RNG (avoids global np.random state)
                flip_mask = self.rng.random(X_rep.shape) < self.prob_var
                X_mut = X_rep ^ flip_mask  # XOR flips True bits where mask is True
                mutated = Population.new("X", X_mut)
                ev.eval(self.problem, mutated)

                F = mutated.get("F").reshape(N, self.n_neighbors)
                best_idx = np.argmin(F, axis=1)
                idx = best_idx + np.arange(N) * self.n_neighbors
                best_neighbors = mutated[idx]

                improve_mask = (best_neighbors.get("F") < my_solutions.get("F")).squeeze(axis=1)
                new_X = np.where(improve_mask[:, None], best_neighbors.get("X"), my_solutions.get("X"))
                new_pop = Population.new("X", new_X)

                mask = improve_mask
                if np.any(mask):
                    ev.eval(self.problem, new_pop[mask])

                new_pop[~mask].set("F", my_solutions[~mask].get("F"))
                my_solutions = new_pop

        # --- MPI Gather ---
        if hasattr(self, 'comm') and self.comm is not None:
            gathered = self.comm.allgather(list(my_solutions))
            all_sols = [s for sublist in gathered for s in sublist]
            return Population.create(*all_sols)
        else:
            return my_solutions