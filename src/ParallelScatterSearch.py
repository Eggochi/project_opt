from mpi4py import MPI
import numpy as np

from pymoo.core.algorithm import Algorithm
from pymoo.core.population import Population
from pymoo.util.display.output import Output
from pymoo.util.display.column import Column
from pymoo.util.optimum import filter_optimum
from collections import Counter


# ============================================================
# Metrics
# ============================================================

class SearchMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.initial_evals = 0
        self.initial_execs = 0
        self.initial_steps = 0

        self.improv_evals = 0
        self.improv_execs = 0
        self.improv_steps = 0

        self.effective_iterations = 0
        self.total_iterations = 0

    def to_dict(self):
        return {
            "initial_improvement": {
                "n_evals": self.initial_evals,
                "n_execs": self.initial_execs,
                "avg_steps": (
                    self.initial_steps / self.initial_execs
                    if self.initial_execs else 0.0
                )
            },
            "improvement": {
                "n_evals": self.improv_evals,
                "n_execs": self.improv_execs,
                "avg_steps": (
                    self.improv_steps / self.improv_execs
                    if self.improv_execs else 0.0
                )
            },
            "effective_iterations": self.effective_iterations,
            "total_iterations": self.total_iterations
        }


# ============================================================
# Output
# ============================================================

class ScatterSearchOutput(Output):
    def __init__(self, comm=None):
        super().__init__()

        self.f_avg = Column("f_avg", width=13)
        self.f_min = Column("f_min", width=13)

        self.columns += [self.f_avg, self.f_min]

        self.comm = comm
        self.rank = comm.Get_rank() if comm is not None else 0

    def update(self, algorithm):
        super().update(algorithm)

        pop = algorithm.pop

        if pop is None or len(pop) == 0:
            self.f_avg.set(None)
            self.f_min.set(None)
            return

        F = pop.get("F")

        if F is None or len(F) == 0:
            self.f_avg.set(None)
            self.f_min.set(None)
            return

        self.f_avg.set(np.mean(F))
        self.f_min.set(np.min(F))


# ============================================================
# Scatter Search
# ============================================================

class ScatterSearch(Algorithm):

    def __init__(
        self,
        subset_generation_method,
        combination_method,
        diversification_method,
        initial_improvement_method,
        improvement_method,
        ReferenceSet,
        repair=None,
        reference_set_size=30,
        solution_pool_size=100,
        diversity_threshold=0.0,
        stagnation_limit=1,
        max_evals=None,
        comm=None,
        **kwargs
    ):
        kwargs.setdefault("output", ScatterSearchOutput(comm=comm))
        super().__init__(**kwargs)

        # MPI
        self.comm = comm
        self.rank = comm.Get_rank() if comm is not None else 0
        self.size = comm.Get_size() if comm is not None else 1
        
        # Termination
        self.max_evals = max_evals

        # Operators
        self.subset_generation_method = subset_generation_method
        self.combination_method = combination_method
        self.diversification_method = diversification_method
        self.initial_improvement_method = initial_improvement_method
        self.improvement_method = improvement_method
        self.ReferenceSet = ReferenceSet
        self.repair = repair

        # Parameters
        self.reference_set_size = reference_set_size
        self.solution_pool_size = int(np.ceil(solution_pool_size / self.size))
        self.diversity_threshold = diversity_threshold
        self.initial_threshold = diversity_threshold
        self.min_threshold = diversity_threshold / 4
        self.stagnation_limit = stagnation_limit

        # State
        self.reference_set = None
        self.n_added = 0
        self.stagnation_counter = 0

        self.pop = None
        self.opt = None

        self.metrics = SearchMetrics()

    # ========================================================
    # MPI Helpers
    # ========================================================

    def _gather_to_root(self, local_items):
        if self.comm is None:
            return list(local_items)

        gathered = self.comm.gather(list(local_items), root=0)

        if self.rank != 0:
            return None

        return [
            item
            for worker_items in gathered
            for item in worker_items
        ]

    def _broadcast_state(self):
        state = None

        if self.comm is None or self.rank == 0:
            state = {
                "reference_set": self.reference_set,
                "n_added": self.n_added,
                "opt": self.opt
            }

        if self.comm is not None:
            state = self.comm.bcast(state, root=0)

        self.reference_set = state["reference_set"]
        self.n_added = state["n_added"]
        self.opt = state["opt"]

    def _refresh_population(self):
        self.pop = Population.create(self.ReferenceSet.RefSet)
        self.opt = filter_optimum(self.pop, least_infeasible=True)

    # ========================================================
    # Operator Propagation
    # ========================================================

    def _propagate_methods(self):
        base_seed = self.seed
        per_rank_seed = (
            base_seed + self.rank
            if base_seed is not None else None
        )

        synchronized_methods = [
            self.ReferenceSet,
            self.subset_generation_method
        ]

        per_rank_methods = [
            self.diversification_method,
            self.initial_improvement_method,
            self.improvement_method,
            self.combination_method,
            self.repair
        ]

        all_methods = synchronized_methods + per_rank_methods

        for method in all_methods:

            if hasattr(method, "set_problem"):
                method.set_problem(self.problem)

            if hasattr(method, "set_evaluator"):
                method.set_evaluator(self.evaluator)

            if hasattr(method, "set_metrics"):
                method.set_metrics(self.metrics)
            
            if hasattr(method, "set_repair"):
                method.set_repair(self.repair)

        for method in synchronized_methods:
            if hasattr(method, "set_seed"):
                method.set_seed(base_seed)

        for method in per_rank_methods:
            if hasattr(method, "set_seed"):
                method.set_seed(per_rank_seed)

    # ========================================================
    # Initialization
    # ========================================================

    def _initialize(self):
        self.metrics.reset()
        self._propagate_methods()

        super()._initialize()

        if (
            hasattr(self.improvement_method, "cache")
            and self.improvement_method.cache is not None
        ):
            self.improvement_method.cache.clear()

        local_pool = self.diversification_method.do(
            self.problem,
            self.solution_pool_size
        )

        if not isinstance(local_pool, Population):
            local_pool = Population.new("X", local_pool)

        local_pool = self.initial_improvement_method.improve_pool(local_pool)

        solution_pool = self._gather_to_root(local_pool)

        if self.rank == 0:
            self.reference_set = self.ReferenceSet.create(
                solution_pool,
                self.reference_set_size,
                diversity_threshold=self.diversity_threshold
            )

            self.n_added = len(self.reference_set[0])
            self._refresh_population()

        self._broadcast_state()

    # ========================================================
    # Main Iteration
    # ========================================================

    def _advance(self, **kwargs):

        if self.rank == 0:
            pairs = self.subset_generation_method.generate(self.reference_set)
            
            # Split pairs into chunks for each process
            scatter_pairs = [
                pairs[i::self.size]
                for i in range(self.size)
            ]
        else:
            scatter_pairs = None

        my_pairs = (
            self.comm.scatter(scatter_pairs, root=0)
            if self.comm is not None
            else scatter_pairs[0]
        )

        my_solutions = self.combination_method.combine(my_pairs)

        if len(my_solutions) != 0:
            my_solutions = self.improvement_method.improve_pool(my_solutions)

        new_solutions = self._gather_to_root(my_solutions)

        if self.rank == 0:
            if len(new_solutions) == 0:
                self.n_added = 0
            else:
                self.reference_set = self.ReferenceSet.update(
                    new_solutions,
                    self.reference_set_size,
                    diversity_threshold=self.diversity_threshold
                )

                self.n_added = len(self.reference_set[0])
                self._refresh_population()

        self._broadcast_state()

        if self.rank == 0:
            self.metrics.total_iterations += 1

            if self.n_added > 0:
                self.metrics.effective_iterations += 1

        self._stagnation_check()

    # ========================================================
    # Restart
    # ========================================================

    def _restart(self):

        # Generar nuevas soluciones mediante el método de diversificación
        local_pool = self.diversification_method.do(self.problem, self.solution_pool_size)
        local_pool = self.initial_improvement_method.improve_pool(local_pool)
        new_pool = self._gather_to_root(local_pool)

        if self.rank == 0:
            elite_size = self.reference_set_size // 2
            elite = self.ReferenceSet.RefSet[:elite_size]
            combined_pool = elite + new_pool
            
            # Crear el nuevo conjunto de referencia evaluando todo el pool combinado
            self.reference_set = self.ReferenceSet.create(
                combined_pool,
                self.reference_set_size,
                diversity_threshold=self.diversity_threshold
            )

            coincidencias = Counter(self.reference_set[0]) & Counter(elite)
            self.n_added = len(self.reference_set[0]) - sum(coincidencias.values())
            #self.n_added = len(self.reference_set[0]) - 1
            self._refresh_population()
        self._broadcast_state()

    # ========================================================
    # Stagnation
    # ========================================================

    def _stagnation_check(self):
        # n_added viene de _broadcast_state, así que TODOS los nodos ven lo mismo
        if self.n_added == 0:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0

        if self.stagnation_counter >= self.stagnation_limit:
            self._restart()
            self.stagnation_counter = 0

    # ========================================================
    # Statistics
    # ========================================================

    def has_next(self):
        local_has_next = super().has_next()
        
        # Check global evaluation limit if specified
        if self.max_evals is not None:
            local_evals = self.evaluator.n_eval if hasattr(self, 'evaluator') else 0
            if self.comm is not None:
                global_evals = self.comm.allreduce(local_evals, op=MPI.SUM)
            else:
                global_evals = local_evals
                
            if global_evals >= self.max_evals:
                local_has_next = False
                
        if self.comm is not None:
            # Sincronizar terminación: si algún proceso debe detenerse (ej. alcanzó el límite de eval), todos deben detenerse.
            # Esto previene deadlocks en llamadas colectivas de MPI.
            global_has_next = self.comm.allreduce(local_has_next, op=MPI.LAND)
            return global_has_next
        return local_has_next

    def get_stats(self):
        local_stats = self.metrics.to_dict()

        if self.comm is None:
            return local_stats

        gathered = self.comm.gather(local_stats, root=0)

        if self.rank != 0:
            return None

        agg = SearchMetrics()

        for s in gathered:
            agg.initial_evals += s["initial_improvement"]["n_evals"]
            agg.initial_execs += s["initial_improvement"]["n_execs"]
            agg.initial_steps += (
                s["initial_improvement"]["avg_steps"]
                * s["initial_improvement"]["n_execs"]
            )

            agg.improv_evals += s["improvement"]["n_evals"]
            agg.improv_execs += s["improvement"]["n_execs"]
            agg.improv_steps += (
                s["improvement"]["avg_steps"]
                * s["improvement"]["n_execs"]
            )

        agg.effective_iterations = max(
            s["effective_iterations"] for s in gathered
        )

        agg.total_iterations = max(
            s["total_iterations"] for s in gathered
        )

        return agg.to_dict()


#==========================
# Sin paralelizacion de la generacion por diversidad
# =========================

class ScatterSearch2(Algorithm):

    def __init__(
        self,
        subset_generation_method,
        combination_method,
        diversification_method,
        initial_improvement_method,
        improvement_method,
        ReferenceSet,
        repair=None,
        reference_set_size=30,
        solution_pool_size=100,
        diversity_threshold=0.0,
        stagnation_limit=1,
        max_evals=None,
        comm=None,
        **kwargs
    ):
        kwargs.setdefault("output", ScatterSearchOutput(comm=comm))
        super().__init__(**kwargs)

        # MPI
        self.comm = comm
        self.rank = comm.Get_rank() if comm is not None else 0
        self.size = comm.Get_size() if comm is not None else 1
        
        # Termination
        self.max_evals = max_evals

        # Operators
        self.subset_generation_method = subset_generation_method
        self.combination_method = combination_method
        self.diversification_method = diversification_method
        self.initial_improvement_method = initial_improvement_method
        self.improvement_method = improvement_method
        self.ReferenceSet = ReferenceSet
        self.repair = repair

        # Parameters
        self.reference_set_size = reference_set_size
        self.solution_pool_size = solution_pool_size 
        self.diversity_threshold = diversity_threshold
        self.initial_threshold = diversity_threshold
        self.min_threshold = diversity_threshold / 4
        self.stagnation_limit = stagnation_limit

        # State
        self.reference_set = None
        self.n_added = 0
        self.stagnation_counter = 0

        self.pop = None
        self.opt = None

        self.metrics = SearchMetrics()

    # ========================================================
    # MPI Helpers
    # ========================================================

    def _gather_to_root(self, local_items):
        if self.comm is None:
            return list(local_items)

        gathered = self.comm.gather(list(local_items), root=0)

        if self.rank != 0:
            return None

        return [
            item
            for worker_items in gathered
            for item in worker_items
        ]

    def _broadcast_state(self):
        state = None

        if self.comm is None or self.rank == 0:
            state = {
                "reference_set": self.reference_set,
                "n_added": self.n_added,
                "opt": self.opt
            }

        if self.comm is not None:
            state = self.comm.bcast(state, root=0)

        self.reference_set = state["reference_set"]
        self.n_added = state["n_added"]
        self.opt = state["opt"]

    def _refresh_population(self):
        self.pop = Population.create(self.ReferenceSet.RefSet)
        self.opt = filter_optimum(self.pop, least_infeasible=True)

    # ========================================================
    # Operator Propagation
    # ========================================================

    def _propagate_methods(self):
        base_seed = self.seed
        per_rank_seed = (
            base_seed + self.rank
            if base_seed is not None else None
        )

        synchronized_methods = [
            self.diversification_method,
            self.ReferenceSet,
            self.subset_generation_method
        ]

        per_rank_methods = [
            self.initial_improvement_method,
            self.improvement_method,
            self.combination_method,
            self.repair
        ]

        all_methods = synchronized_methods + per_rank_methods

        for method in all_methods:

            if hasattr(method, "set_problem"):
                method.set_problem(self.problem)

            if hasattr(method, "set_evaluator"):
                method.set_evaluator(self.evaluator)

            if hasattr(method, "set_metrics"):
                method.set_metrics(self.metrics)
            
            if hasattr(method, "set_repair"):
                method.set_repair(self.repair)

        for method in synchronized_methods:
            if hasattr(method, "set_seed"):
                method.set_seed(base_seed)

        for method in per_rank_methods:
            if hasattr(method, "set_seed"):
                method.set_seed(per_rank_seed)

    # ========================================================
    # Initialization
    # ========================================================

    def _initialize(self):
        self.metrics.reset()
        self._propagate_methods()

        super()._initialize()

        # Limpiar cache si existe
        if (hasattr(self.improvement_method, "cache") 
            and self.improvement_method.cache is not None):
            self.improvement_method.cache.clear()

        solution_pool = None

        # 1. Generación de Diversificación (Solo Rango 0)
        if self.rank == 0:
            solution_pool = self.diversification_method.do(
                self.problem,
                self.solution_pool_size
            )
            if not isinstance(solution_pool, Population):
                solution_pool = Population.new("X", solution_pool)

            # Dividir la población en fragmentos para cada proceso
            # Esto crea una lista de longitud self.size
            scatter_chunks = [
                solution_pool[i::self.size] for i in range(self.size)
            ]
        else:
            scatter_chunks = None

        # 2. Distribuir fragmentos entre procesos
        my_chunk = (
            self.comm.scatter(scatter_chunks, root=0)
            if self.comm is not None
            else solution_pool
        )

        # 3. Mejora Local (Paralela)
        # Cada proceso mejora su propia parte
        my_chunk = self.initial_improvement_method.improve_pool(my_chunk)

        # 4. Reunir todo de vuelta en el Rango 0
        if self.comm is not None:
            improved_chunks = self.comm.gather(my_chunk, root=0)
            if self.rank == 0:
                # Combinar los fragmentos mejorados en una sola población
                if len(improved_chunks) == 1:
                    solution_pool = improved_chunks[0]
                else:
                    solution_pool = Population.merge(*improved_chunks)
        else:
            solution_pool = my_chunk

        # 5. Creación del Reference Set (Solo Rango 0 con datos mejorados)
        if self.rank == 0:
            self.reference_set = self.ReferenceSet.create(
                solution_pool,
                self.reference_set_size,
                diversity_threshold=self.diversity_threshold
            )

            self.n_added = len(self.reference_set[0])
            self._refresh_population()

        # 6. Sincronizar el estado final a todos los procesos
        self._broadcast_state()
    # ========================================================
    # Main Iteration
    # ========================================================

    def _advance(self, **kwargs):

        if self.rank == 0:
            pairs = self.subset_generation_method.generate(self.reference_set)
            
            # Split pairs into chunks for each process
            scatter_pairs = [
                pairs[i::self.size]
                for i in range(self.size)
            ]
        else:
            scatter_pairs = None

        my_pairs = (
            self.comm.scatter(scatter_pairs, root=0)
            if self.comm is not None
            else scatter_pairs[0]
        )

        my_solutions = self.combination_method.combine(my_pairs)

        if len(my_solutions) != 0:
            my_solutions = self.improvement_method.improve_pool(my_solutions)

        new_solutions = self._gather_to_root(my_solutions)

        if self.rank == 0:
            if len(new_solutions) == 0:
                self.n_added = 0
            else:
                self.reference_set = self.ReferenceSet.update(
                    new_solutions,
                    self.reference_set_size,
                    diversity_threshold=self.diversity_threshold
                )

                self.n_added = len(self.reference_set[0])
                self._refresh_population()

        self._broadcast_state()

        if self.rank == 0:
            self.metrics.total_iterations += 1

            if self.n_added > 0:
                self.metrics.effective_iterations += 1

        self._stagnation_check()

    # ========================================================
    # Restart
    # ========================================================

    def _restart(self):

        solution_pool = None

        # 1. Generación de Diversificación (Solo Rango 0)
        if self.rank == 0:
            solution_pool = self.diversification_method.do(
                self.problem,
                self.solution_pool_size
            )
            if not isinstance(solution_pool, Population):
                solution_pool = Population.new("X", solution_pool)

            scatter_chunks = [
                solution_pool[i::self.size] for i in range(self.size)
            ]
        else:
            scatter_chunks = None

        # 2. Distribuir fragmentos
        my_chunk = (
            self.comm.scatter(scatter_chunks, root=0)
            if self.comm is not None
            else solution_pool
        )

        # 3. Mejora Local (Paralela)
        my_chunk = self.initial_improvement_method.improve_pool(my_chunk)

        # 4. Reunir
        if self.comm is not None:
            improved_chunks = self.comm.gather(my_chunk, root=0)
            if self.rank == 0:
                if len(improved_chunks) == 1:
                    new_pool = improved_chunks[0]
                else:
                    new_pool = Population.merge(*improved_chunks)
        else:
            new_pool = my_chunk

        if self.rank == 0:
            elite_size = self.reference_set_size // 2
            elite = self.ReferenceSet.RefSet[:elite_size]
            
            # Convirtiendo a listas para poder sumar
            combined_pool = list(elite) + list(new_pool)
            
            # Crear el nuevo conjunto de referencia evaluando todo el pool combinado
            self.reference_set = self.ReferenceSet.create(
                combined_pool,
                self.reference_set_size,
                diversity_threshold=self.diversity_threshold
            )

            coincidencias = Counter(self.reference_set[0]) & Counter(elite)
            self.n_added = len(self.reference_set[0]) - sum(coincidencias.values())
            self._refresh_population()
        self._broadcast_state()

    # ========================================================
    # Stagnation
    # ========================================================

    def _stagnation_check(self):
        if self.n_added == 0:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0

        if self.stagnation_counter >= self.stagnation_limit:
            self._restart()
            self.stagnation_counter = 0

    # ========================================================
    # Statistics
    # ========================================================

    def has_next(self):
        local_has_next = super().has_next()
        
        # Check global evaluation limit if specified
        if self.max_evals is not None:
            local_evals = self.evaluator.n_eval if hasattr(self, 'evaluator') else 0
            if self.comm is not None:
                global_evals = self.comm.allreduce(local_evals, op=MPI.SUM)
            else:
                global_evals = local_evals
                
            if global_evals >= self.max_evals:
                local_has_next = False
                
        if self.comm is not None:
            # Sincronizar terminación: si algún proceso debe detenerse (ej. alcanzó el límite de eval), todos deben detenerse.
            # Esto previene deadlocks en llamadas colectivas de MPI.
            global_has_next = self.comm.allreduce(local_has_next, op=MPI.LAND)
            return global_has_next
        return local_has_next

    def get_stats(self):
        local_stats = self.metrics.to_dict()

        if self.comm is None:
            return local_stats

        gathered = self.comm.gather(local_stats, root=0)

        if self.rank != 0:
            return None

        agg = SearchMetrics()

        for s in gathered:
            agg.initial_evals += s["initial_improvement"]["n_evals"]
            agg.initial_execs += s["initial_improvement"]["n_execs"]
            agg.initial_steps += (
                s["initial_improvement"]["avg_steps"]
                * s["initial_improvement"]["n_execs"]
            )

            agg.improv_evals += s["improvement"]["n_evals"]
            agg.improv_execs += s["improvement"]["n_execs"]
            agg.improv_steps += (
                s["improvement"]["avg_steps"]
                * s["improvement"]["n_execs"]
            )

        agg.effective_iterations = max(
            s["effective_iterations"] for s in gathered
        )

        agg.total_iterations = max(
            s["total_iterations"] for s in gathered
        )

        return agg.to_dict()