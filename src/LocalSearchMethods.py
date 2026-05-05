import numpy as np
import math 
import itertools
from pymoo.core.population import Population


class LocalSearch:

    def __init__(
        self,
        neighborhood,
        method="best_improvement",
        max_steps=1,
        use_cache=False,
        metric_prefix="improv"
    ):

        self.neighborhood = neighborhood
        self.method = method
        self.max_steps = max_steps
        self.use_cache = use_cache
        self.metric_prefix = metric_prefix

        self.problem = None
        self.repair = None
        self.evaluator = None
        self.metrics = None

        self.cache = {} if use_cache else None

    # =====================================================
    # Injection
    # =====================================================

    def set_problem(self, problem):
        self.problem = problem

    def set_repair(self, repair):
        self.repair = repair

    def set_evaluator(self, evaluator):
        self.evaluator = evaluator

    def set_metrics(self, metrics):
        self.metrics = metrics

    def set_seed(self, seed):
        self.neighborhood.set_seed(seed)

    # =====================================================
    # Evaluation
    # =====================================================

    def _get_evaluator(self):
        from pymoo.core.evaluator import Evaluator
        return self.evaluator if self.evaluator else Evaluator()

    def _register_eval(self, n):
        if self.metrics is None:
            return
        if self.metric_prefix == "initial":
            self.metrics.initial_evals += n
        else:
            self.metrics.improv_evals += n

    def _register_exec(self):
        if self.metrics is None:
            return
        if self.metric_prefix == "initial":
            self.metrics.initial_execs += 1
        else:
            self.metrics.improv_execs += 1

    def _register_steps(self, n):
        if self.metrics is None:
            return
        if self.metric_prefix == "initial":
            self.metrics.initial_steps += n
        else:
            self.metrics.improv_steps += n

    def _evaluate_one(self, x):

        key = x.tobytes()

        if self.use_cache and key in self.cache:
            return self.cache[key]

        pop = Population.new("X", np.array([x]))
        self._get_evaluator().eval(self.problem, pop)

        f = float(pop.get("F").flat[0])

        self._register_eval(1)

        if self.use_cache:
            self.cache[key] = f

        return f

    def _evaluate_batch(self, X):

        pop = Population.new("X", X)
        self._get_evaluator().eval(self.problem, pop)

        F = pop.get("F").flatten()

        self._register_eval(len(X))

        if self.use_cache:
            for x, f in zip(X, F):
                self.cache[x.tobytes()] = f

        return F

    def _ensure_valid_fitness(self, solutions):
        """
        Repairs and evaluates only solutions with missing/invalid fitness.

        Returns
        -------
        Population
            Same population with valid F for all individuals.
        """

        if len(solutions) == 0:
            return solutions

        F_existing = solutions.get("F")

        # ------------------------------------------
        # Detect invalid / missing fitness
        # ------------------------------------------
        if F_existing is None:
            invalid_mask = np.ones(len(solutions), dtype=bool)

        else:
            invalid_mask = np.array([
                f is None
                or np.asarray(f).size == 0
                or np.any(np.isnan(np.asarray(f, dtype=float)))
                for f in F_existing
            ])

        # ------------------------------------------
        # Repair / Evaluate only invalid individuals
        # ------------------------------------------
        if np.any(invalid_mask):

            invalid_pop = solutions[invalid_mask]

            if self.repair is not None:
                invalid_pop = self.repair.do(self.problem, invalid_pop)

            F_new = self._evaluate_batch(invalid_pop.get("X"))

            invalid_pop.set("F", F_new.reshape(-1, 1))

            solutions[invalid_mask] = invalid_pop

        return solutions

    # =====================================================
    # Improvement Policies
    # =====================================================

    def _first_improvement(self, x, f_current, X_neighbors):

        for xn in X_neighbors:
            fn = self._evaluate_one(xn)

            if fn < f_current:
                return xn.copy(), fn, True

        return x, f_current, False

    def _best_improvement(self, x, f_current, X_neighbors):

        F_neighbors = self._evaluate_batch(X_neighbors)

        best_idx = np.argmin(F_neighbors)

        if F_neighbors[best_idx] < f_current:
            return X_neighbors[best_idx].copy(), F_neighbors[best_idx], True

        return x, f_current, False

    # =====================================================
    # Main
    # =====================================================

    def improve_pool(self, solutions):

        if self.problem is None:
            raise ValueError("Problem not set")


        if not isinstance(solutions, Population):
            solutions = Population.create(*solutions)

        solutions = self._ensure_valid_fitness(solutions)    

        improved = []

    def improve_pool(self, solutions):

        if not isinstance(solutions, Population):
            solutions = Population.create(*solutions)

        if len(solutions) == 0:
            return solutions

        solutions = self._ensure_valid_fitness(solutions)

        X = solutions.get("X").copy()
        F = solutions.get("F").flatten().copy()

        N = len(solutions)

        active = np.ones(N, dtype=bool)
        steps = np.zeros(N, dtype=int)

        while np.any(active):

            if self.max_steps is not None:
                active &= (steps < self.max_steps)

            if not np.any(active):
                break

            active_idx = np.where(active)[0]

            # -----------------------------------------
            # Generate all neighborhoods
            # -----------------------------------------
            neighbor_blocks = []
            owner = []

            for i in active_idx:
                Xi_neighbors = self.neighborhood.generate(X[i])

                if self.repair is not None:
                    pop = Population.new("X", Xi_neighbors)
                    Xi_neighbors = self.repair.do(
                        self.problem, pop
                    ).get("X")

                Xi_neighbors = np.unique(Xi_neighbors, axis=0)

                neighbor_blocks.append(Xi_neighbors)
                owner.extend([i] * len(Xi_neighbors))

            if not neighbor_blocks:
                break

            X_neighbors_all = np.vstack(neighbor_blocks)
            owner = np.asarray(owner)

            F_neighbors_all = self._evaluate_batch(X_neighbors_all)

            # -----------------------------------------
            # Select Improvements
            # -----------------------------------------
            improved_this_round = np.zeros(N, dtype=bool)

            start = 0

            for i, Xi_neighbors in zip(active_idx, neighbor_blocks):

                m = len(Xi_neighbors)

                Xi_F = F_neighbors_all[start:start+m]
                start += m

                if self.method == "best_improvement":

                    best_idx = np.argmin(Xi_F)

                    if Xi_F[best_idx] < F[i]:
                        X[i] = Xi_neighbors[best_idx]
                        F[i] = Xi_F[best_idx]
                        improved_this_round[i] = True

                elif self.method == "first_improvement":

                    better = np.where(Xi_F < F[i])[0]

                    if len(better) > 0:
                        j = better[0]
                        X[i] = Xi_neighbors[j]
                        F[i] = Xi_F[j]
                        improved_this_round[i] = True

                else:
                    raise ValueError("Invalid method")

            steps[improved_this_round] += 1
            active &= improved_this_round

        solutions.set("X", X)
        solutions.set("F", F.reshape(-1, 1))

        return solutions
        
class NeighborhoodGenerator:

    def generate(self, x, rng):
        raise NotImplementedError

class BitFlipMutationNeighborhood(NeighborhoodGenerator):

    def __init__(self, sample=5, prob_var=0.05):
        self.sample = sample
        self.prob_var = prob_var
        self.rng = np.random.default_rng()

    def set_seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def generate(self, x):

        X = np.tile(x, (self.sample, 1))

        flip_mask = self.rng.random(X.shape) < self.prob_var

        return np.logical_xor(X, flip_mask)

class SingleBitFlipNeighborhood(NeighborhoodGenerator):

    def __init__(self, sample=None):
        self.sample = sample
        self.rng = np.random.default_rng()

    def set_seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def generate(self, x):

        n = len(x)

        if self.sample is None:
            indices = self.rng.permutation(n)
        else:
            k = min(self.sample, n)
            indices = self.rng.choice(n, k, replace=False)

        X = np.tile(x, (len(indices), 1))
        X[np.arange(len(indices)), indices] ^= True

        return X


class SingleBitActivateNeighborhood(NeighborhoodGenerator):

    def __init__(self, sample=None):
        self.sample = sample
        self.rng = np.random.default_rng()

    def set_seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def generate(self, x):

        n = len(x)

        deactivated_indices = np.flatnonzero(~x)

        if self.sample is None:
            indices = self.rng.permutation(deactivated_indices)
        else:
            k = min(self.sample, len(deactivated_indices))
            indices = self.rng.choice(deactivated_indices, k, replace=False)

        X = np.tile(x, (len(indices), 1))
        X[np.arange(len(indices)), indices] ^= True

        return X

class KnapsackNeighborhood(NeighborhoodGenerator):

    def __init__(
        self,
        weights,
        capacity,
        use_swap=True,
        max_swaps=None,
        shuffle=True
    ):
        self.weights = np.asarray(weights)
        self.capacity = capacity
        self.use_swap = use_swap
        self.max_swaps = max_swaps
        self.shuffle = shuffle

        self.rng = np.random.default_rng()

    def set_seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def generate(self, x):

        x = np.asarray(x, dtype=bool)

        n = len(x)
        current_weight = np.dot(x, self.weights)

        selected = np.flatnonzero(x)
        unselected = np.flatnonzero(~x)

        neighbors = []

        # =====================================================
        # DROP moves
        # =====================================================
        if len(selected):
            drop_neighbors = np.tile(x, (len(selected), 1))
            drop_neighbors[np.arange(len(selected)), selected] = False
            neighbors.append(drop_neighbors)

        # =====================================================
        # ADD moves
        # =====================================================
        if len(unselected):
            feasible_add = unselected[
                current_weight + self.weights[unselected] <= self.capacity
            ]

            if len(feasible_add):
                add_neighbors = np.tile(x, (len(feasible_add), 1))
                add_neighbors[np.arange(len(feasible_add)), feasible_add] = True
                neighbors.append(add_neighbors)

        # =====================================================
        # SWAP moves (VECTORIZE)
        # =====================================================
        if self.use_swap and len(selected) and len(unselected):

            # Matrix of resulting weights for every swap (i,j)
            swap_weights = (
                current_weight
                - self.weights[selected][:, None]
                + self.weights[unselected][None, :]
            )

            feasible_mask = swap_weights <= self.capacity

            sel_idx, unsel_idx = np.where(feasible_mask)

            if len(sel_idx):

                if self.max_swaps is not None and len(sel_idx) > self.max_swaps:
                    chosen = self.rng.choice(
                        len(sel_idx),
                        self.max_swaps,
                        replace=False
                    )
                    sel_idx = sel_idx[chosen]
                    unsel_idx = unsel_idx[chosen]

                swap_neighbors = np.tile(x, (len(sel_idx), 1))

                swap_neighbors[
                    np.arange(len(sel_idx)),
                    selected[sel_idx]
                ] = False

                swap_neighbors[
                    np.arange(len(sel_idx)),
                    unselected[unsel_idx]
                ] = True

                neighbors.append(swap_neighbors)

        # =====================================================
        # Final Assembly
        # =====================================================
        if not neighbors:
            return np.empty((0, n), dtype=bool)

        X = np.vstack(neighbors)

        if self.shuffle:
            self.rng.shuffle(X)

        return X

class MultipleBitActivateNeighborhood:

    def __init__(self, n_bits=2, sample=None):
        self.n_bits = n_bits
        self.sample = sample
        self.rng = np.random.default_rng()

    def set_seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def generate(self, x):

        x = np.asarray(x, dtype=bool)

        inactive = np.flatnonzero(~x)
        m = len(inactive)

        if m < self.n_bits:
            return np.empty((0, len(x)), dtype=bool)

        max_combos = math.comb(m, self.n_bits)

        # --------------------------------------------------
        # Full Enumeration
        # --------------------------------------------------
        if self.sample is None or self.sample >= max_combos:

            combos = np.array(
                list(itertools.combinations(inactive, self.n_bits)),
                dtype=int
            )

        # --------------------------------------------------
        # Unique Random Sampling WITHOUT replacement
        # --------------------------------------------------
        else:

            combos_set = set()

            while len(combos_set) < self.sample:

                combo = tuple(sorted(
                    self.rng.choice(
                        inactive,
                        self.n_bits,
                        replace=False
                    )
                ))

                combos_set.add(combo)

            combos = np.array(list(combos_set), dtype=int)

        # --------------------------------------------------
        # Build Neighbors
        # --------------------------------------------------
        X = np.tile(x, (len(combos), 1))

        row_idx = np.repeat(np.arange(len(combos)), self.n_bits)
        col_idx = combos.reshape(-1)

        X[row_idx, col_idx] = True

        return X