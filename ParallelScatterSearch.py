from mpi4py import MPI

from pymoo.core.algorithm import Algorithm
from pymoo.core.evaluator import Evaluator
from pymoo.util.display.output import Output
from pymoo.util.display.column import Column
from scipy.spatial.distance import cdist
import numpy as np
from itertools import combinations, product  
from pymoo.core.population import Population
import random
from pymoo.core.individual import Individual
from pymoo.util.optimum import filter_optimum


class ScatterSearchOutput(Output):
    """Verbose display for ScatterSearch — mirrors GA's n_gen | n_eval | f_avg | f_min table."""

    def __init__(self, comm=None):
        super().__init__()
        self.f_avg = Column("f_avg", width=13)
        self.f_min = Column("f_min", width=13)
        self.columns += [self.f_avg, self.f_min]

        #parallel thingys
        self.comm = comm
        self.rank = comm.Get_rank() if comm is not None else 0
        self.size = comm.Get_size() if comm is not None else 1


    def update(self, algorithm):
        super().update(algorithm)  # updates n_gen and n_eval columns
        pop = algorithm.pop
        if pop is None or len(pop) == 0:
            self.f_avg.set(None)
            self.f_min.set(None)
            return
        F = pop.get("F")
        if F is not None and len(F) > 0:
            self.f_avg.set(np.mean(F))
            self.f_min.set(np.min(F))
        else:
            self.f_avg.set(None)
            self.f_min.set(None)

class ScatterSearch(Algorithm):
    def __init__(self, subset_generation_method, combination_method, diversification_method, initial_improvement_method, improvement_method, ReferenceSet,
                 reference_set_size=10, solution_pool_size=100, diversity_threshold=0.0, stagnation_limit=5, comm=None, **kwargs):
        # Set our custom display before calling super().__init__ so pymoo picks it up
        kwargs.setdefault("output", ScatterSearchOutput(comm=comm))
        super().__init__(**kwargs)

        #parallel thingys
        self.comm = comm
        self.rank = comm.Get_rank() if comm is not None else 0
        self.size = comm.Get_size() if comm is not None else 1

        self.subset_generation_method = subset_generation_method
        self.combination_method = combination_method
        self.diversification_method = diversification_method
        self.initial_improvement_method = initial_improvement_method
        self.improvement_method = improvement_method
        self.ReferenceSet = ReferenceSet
        self.reference_set_size = reference_set_size
        self.solution_pool_size = solution_pool_size // self.size
        self.diversity_threshold = diversity_threshold
        self.stagnation_limit = stagnation_limit
        self.stagnation_counter = 0
        self.n_added = 0
    
    def _initialize(self):
        super()._initialize()

        self._propagate_problem()
        # Clear stale cache entries — they were keyed to the previous problem.
        if hasattr(self.improvement_method, 'cache') and self.improvement_method.cache is not None:
            self.improvement_method.cache.clear()

        # Propagate the algorithm's own Evaluator so n_eval is tracked in verbose output
        for method in (self.initial_improvement_method, self.improvement_method):
            if hasattr(method, 'set_evaluator'):
                method.set_evaluator(self.evaluator)

        #Diversification Method
        #Generate initial solution pool using pymoo's do()
        solution_pool = self.diversification_method.do(self.problem, self.solution_pool_size)
        
        #Improvement Method
        solution_pool = self.initial_improvement_method.improve_pool(solution_pool)
        
        #Reference Set Creation
        if self.comm is not None:
            # Gather all shards to all processors so everyone can build the RefSet identically
            gather_solution_pool = self.comm.allgather(list(solution_pool))
            # Flatten the list of lists
            solution_pool = [s for sublist in gather_solution_pool for s in sublist]
        
        self.reference_set = self.ReferenceSet.create(solution_pool, self.reference_set_size, diversity_threshold=self.diversity_threshold)
        self.n_added = len(self.reference_set[0])
        
        # Pymoo telemetry hook
        self.pop = Population.create(*self.reference_set[0])
        
        # In Pymoo 0.6, self.opt must be assigned explicitly or filtered
        from pymoo.util.optimum import filter_optimum
        self.opt = filter_optimum(self.pop, least_infeasible=True)
    
    def _advance(self, **kwargs):
        # Generate subsets (pairs) using the configured subset generation scheme
        # All ranks generate the same pairs (logic is fast)
        pairs = self.subset_generation_method.generate(self.reference_set)
        
        # Apply combination method. 
        # Internally, this will split the workload across MPI ranks and allgather results.
        new_solutions = self.combination_method.combine(pairs)
        
        # Improvement Method
        # Internally, this will split the workload across MPI ranks and allgather results.
        new_solutions = self.improvement_method.improve_pool(new_solutions)
        
        # Update Reference Set
        # Everyone has the same new_solutions (due to allgather inside methods), 
        # so they can all update locally and stay in sync.
        self.reference_set = self.ReferenceSet.update(new_solutions, self.reference_set_size, diversity_threshold=self.diversity_threshold)
        
        # Calculate how many were added (for telemetry)
        self.n_added = len(self.reference_set[0])
        
        # Check for stagnation
        
        
        self._stagnation_check()
        
        # Pymoo telemetry hook: it reads the population from self.pop
        # We use the full RefSet to ensure self.opt is correctly updated
        self.pop = Population.create(*self.ReferenceSet.RefSet)
        self.opt = filter_optimum(self.pop, least_infeasible=True)
    
    def _stagnation_check(self):
        #if self.n_added == 0:
        #    self.stagnation_counter += 1
        #else:
        #    self.stagnation_counter = 0
        #
        #if self.stagnation_counter >= self.stagnation_limit:
        #    self._restart()
        #    self.stagnation_counter = 0
        if self.n_added == 0:
            self._restart()

    def _restart(self):
        # 1. Conservar el mejor absoluto
        best_sol = self.opt[0] if self.opt is not None else self.reference_set[0][0]
        
        # 2. Generar un nuevo pool diverso
        new_pool = self.diversification_method.do(self.problem, self.solution_pool_size - 1)
        new_pool = self.initial_improvement_method.improve_pool(new_pool)
        
        # 3. Combinar el mejor con el nuevo pool
        combined_pool = [best_sol] + list(new_pool)
        
        # 4. Re-crear el Reference Set desde cero para asegurar diversidad
        self.ReferenceSet.create(combined_pool, self.reference_set_size, diversity_threshold=0.0)
        self.reference_set = (self.ReferenceSet.RefSet, [])  # (new_solutions, old_solutions)
        self.n_added = self.reference_set_size  # Mark as "fully updated" for telemetry

    def _propagate_problem(self):
        for method in [self.subset_generation_method, self.combination_method, self.diversification_method, self.initial_improvement_method, self.improvement_method, self.ReferenceSet]:
            if hasattr(method, 'set_problem'):
                method.set_problem(self.problem)
            if hasattr(method, 'set_evaluator'):
                method.set_evaluator(self.evaluator)
            if hasattr(method, 'set_comm'):
                method.set_comm(self.comm)
            
        





            

            
    
        





'''
Ideas

1. Usar un metodo de diversificacion mas robusto
2. no hacer el calculo de la funcion objetivo si no se alcanza el umbral de diversidad
3. subset generation sistematico
4. change methods para no necesitar pedir el problema hasta el paso de improvement
5. Reinicio de poblacion despues de no generar nuevas soluciones
6. Paro despues de no encontrar mejora en 3 iteraciones reinicios de poblacion
7. Contador de evaluaciones de la funcion objetivo y cuanto se llego a obtimizar
8. grafica de convergencia tabulacion de cada 100 evaluaciones de la funcion objetivo cual fue el fitness
9. despues de 100 a donde va...

busqueda local... dependiendo del tamano de la cadena (probabilidad de cambio de 5%) hacer un bit flip)
hacer el movimiento de cambio n(5) veces y ver si mejora y me quedo con el mejor y ese lo integro
'''