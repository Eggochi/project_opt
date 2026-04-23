from problemas import NKLandscape, QUBO
from RefSetMethods import ReferenceSet
from LocalSearchMethods import LocalSearchImprovement, LocalSearch_BitFlipMutation
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.pntx import TwoPointCrossover, SinglePointCrossover
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.optimize import minimize
from CombinationMethods import PymooCrossoverCombination, TournamentSubsetGeneration, ExhaustiveSubsetGeneration, TournamentSubsetGeneration2, PathRelinking_RCL, BinaryTournamentSubsetGeneration
from population import FrequencyBinaryDiversification,FrequencyBinaryDiversification2
from ScatterSearch import ScatterSearch

from mpi4py import MPI
import numpy as np
import time

def main():
    # Inicializar MPI primero
    # QUBO — Solucion trivial
    n = 100
    
    # 1. LA MATRIZ DEL PROBLEMA DEBE SER IDÉNTICA PARA TODOS
    # Forzamos la semilla a 42 en todos los nodos antes de crear Q
    np.random.seed(42)
    Q = np.random.randint(-4, 9, size=(n, n))
    Q = (Q + Q.T) // 2
    for i in range(0, n, 2):
        for j in range(0, n, 2):
            Q[i, j] = -6
            Q[j, i] = -6
        Q[i, i] = -10

    problem_qubo = QUBO(Q)

    probflip = 2 / n

    combination_method = PymooCrossoverCombination(TwoPointCrossover())
    diversification_method = FrequencyBinaryDiversification2()
    starting_improv = LocalSearch_BitFlipMutation(prob_var=probflip, n_neighbors=2, max_steps=2)
    improvement_method = LocalSearch_BitFlipMutation(prob_var=probflip, n_neighbors=3, max_steps=2)
    
    time_start = time.time()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # Initialize the algorithm
    algorithm = ScatterSearch(
        subset_generation_method=BinaryTournamentSubsetGeneration(),
        combination_method=combination_method,
        diversification_method=diversification_method,
        initial_improvement_method=starting_improv,
        improvement_method=improvement_method,
        ReferenceSet=ReferenceSet('Hamming'),
        reference_set_size=10,
        solution_pool_size=100,
        diversity_threshold=probflip * 2,
        comm=comm
    )        

    # 2. SEMILLA DISTINTA PARA CADA NODO EN EL ALGORITMO
    # Si no hacemos esto, el paralelismo es inútil porque todos exploran lo mismo
    my_seed = 42 + rank

    if rank == 0:
        print(f"Iniciando Scatter Search con {size} procesos...")

    # Sincronizamos procesos antes de medir el tiempo real
    comm.barrier()
    
    # Solo activamos verbose en rank 0 para no ensuciar la consola
    res = minimize(
        problem_qubo, 
        algorithm, 
        ('n_gen', 100), 
        seed=my_seed, 
        verbose=(rank == 0) 
    )
    
    comm.barrier()
    time_end = time.time()

    # 3. SOLO EL MAESTRO IMPRIME LOS RESULTADOS

    if rank == 0:
        print(f"ScatterSearch time: {time_end - time_start:.4f} segundos")
        print("\nMejor solución encontrada (X):")
        print(res.X)
        print("\nValor objetivo (F):")
        print(res.F)


    algorithm = GA(
        pop_size=100,
        sampling=BinaryRandomSampling(),
        crossover=TwoPointCrossover(),
        mutation=BitflipMutation(prob_var=probflip),
        eliminate_duplicates=True
    )

    time_start = time.time()
    res = minimize(
        problem_qubo,
        algorithm,
        ('n_gen', 100),
        seed=42,
        verbose=True
    )
    time_end = time.time()

    if rank == 0:
        print(f"Genetic Algorithm time: {time_end - time_start:.4f} segundos")
        print("\nMejor solución encontrada (X):")
        print(res.X)
        print("\nValor objetivo (F):")
        print(res.F)


if __name__ == "__main__":
    main()