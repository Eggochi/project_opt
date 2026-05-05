from pymoo.core.individual import Individual
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.pntx import TwoPointCrossover, SinglePointCrossover
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.problems.single.knapsack import create_random_knapsack_problem
from pymoo.optimize import minimize


from src.RefSetMethods import ReferenceSet
from src.LocalSearchMethods import (LocalSearch,
                                 BitFlipMutationNeighborhood,
                                 SingleBitFlipNeighborhood,
                                 KnapsackNeighborhood,
                                 SingleBitActivateNeighborhood,
                                 MultipleBitActivateNeighborhood)
from src.CombinationMethods import PymooCrossoverCombination, ExhaustiveSubsetGeneration, PathRelinking_RCL, BinaryTournamentSubsetGeneration
from src.population import FrequencyBinaryDiversification
from src.ParallelScatterSearch import ScatterSearch
from src.config import config
from src.problemas import NKLandscape, QUBO, RepairKnapsack, RandomRepairKnapsack, RepairKnapsack2


from mpi4py import MPI
import numpy as np
import time

def main():
    # Inicializar MPI primero

    n = config.get("problem", "n", default=100)
    seed = config.get("problem", "seed", default=42)
    
  
    #Create knapsack problem
    problem = create_random_knapsack_problem(n, seed=seed)
    print(problem.C)
    print(np.mean(problem.W), np.std(problem.W))
    print(np.sum(problem.W))
    print(np.mean(problem.P), np.std(problem.P))


    subset_gen = ExhaustiveSubsetGeneration(50)
    neighborhood_generator = MultipleBitActivateNeighborhood(n_bits=2,sample=5)
    combination_method = PymooCrossoverCombination(SinglePointCrossover())
    diversification_method = BinaryRandomSampling()
    starting_improv = LocalSearch(neighborhood=BitFlipMutationNeighborhood(prob_var=1/n, sample=5),
                                  max_steps=1, 
                                  method="best_improvement")
    
    improvement_method = LocalSearch(neighborhood=neighborhood_generator,
                                  max_steps=3, 
                                  method="best_improvement")
    
    time_start = time.time()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # Initialize the algorithm
    algorithm = ScatterSearch(
        subset_generation_method=subset_gen,
        combination_method=combination_method,
        diversification_method=diversification_method,
        initial_improvement_method=starting_improv,
        improvement_method=improvement_method,
        ReferenceSet=ReferenceSet('Hamming'),
        repair=RepairKnapsack2(alpha=2),
        reference_set_size=config.get("algorithm", "reference_set_size", default=20),
        solution_pool_size=config.get("algorithm", "solution_pool_size", default=100),
        diversity_threshold=0,
        comm=comm
    )        

    if rank == 0:
        print(f"Iniciando Scatter Search con {size} procesos...")

    # Sincronizamos procesos antes de medir el tiempo real
    comm.barrier()
    
    # seed=42 is the base seed for all ranks.
    # _propagate_methods gives diversification/subset-gen seed=42 (identical across ranks)
    # and improvement methods seed=42+rank (different per rank, so local search explores
    # different neighborhoods in parallel).
    res = minimize(
        problem, 
        algorithm, 
        ('n_eval', config.get("algorithm", "n_gen", default=300)*200), 
        seed=seed, 
        verbose=(rank == 0) 
    )
    
    comm.barrier()
    time_end = time.time()

    # 3. SOLO EL MAESTRO IMPRIME LOS RESULTADOS

    if rank == 0:

        print(f"ScatterSearch time: {time_end - time_start:.4f} segundos")
        print("\nValor objetivo (F):")
        print(res.F)
        print("\nRestriccion:")
        print(res.G)



if __name__ == "__main__":
    main()