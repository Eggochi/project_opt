import os
import sys
import time
import numpy as np
from mpi4py import MPI
from pymoo.optimize import minimize

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from problemas import QUBO
from RefSetMethods import ReferenceSet
from LocalSearchMethods import LocalSearchFirstImprovement, LocalSearch_BitFlipMutation
from CombinationMethods import ExhaustiveSubsetGeneration, PymooCrossoverCombination
from population import FrequencyBinaryDiversification
from ParallelScatterSearch import ScatterSearch
from pymoo.operators.crossover.pntx import SinglePointCrossover

def create_qubo_problem(n, seed):
    np.random.seed(seed)
    Q = np.random.randint(-10, 10, size=(n, n))
    Q = (Q + Q.T) // 2
    return QUBO(Q)

def benchmark_kernel(n_problem, n_gen=20):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    problem = create_qubo_problem(n_problem, seed=42)

    probflip = 2 / n_problem
    neighbor_sample = max(1, n_problem // 10)

    algorithm = ScatterSearch(
        subset_generation_method=ExhaustiveSubsetGeneration(),
        combination_method=PymooCrossoverCombination(SinglePointCrossover()),
        diversification_method=FrequencyBinaryDiversification(),
        initial_improvement_method=LocalSearch_BitFlipMutation(
            prob_var=probflip,
            n_neighbors=neighbor_sample,
            max_steps=1,
            metric_prefix="initial"
        ),
        improvement_method=LocalSearchFirstImprovement(
            neighbor_sample=neighbor_sample,
            max_steps=3,
            metric_prefix="improv"
        ),
        ReferenceSet=ReferenceSet("Hamming"),
        reference_set_size=12,
        solution_pool_size=128,
        diversity_threshold=probflip * 1.5,
        comm=comm
    )

    if rank == 0:
        print(f"[Kernel Benchmark] n={n_problem}, procs={size}, gens={n_gen}")

    comm.Barrier()
    t0 = MPI.Wtime()

    minimize(
        problem,
        algorithm,
        ("n_gen", n_gen),
        seed=42,
        verbose=False
    )

    comm.Barrier()
    t1 = MPI.Wtime()

    stats = algorithm.get_stats()

    # =========================
    # GLOBAL TIMING BREAKDOWN
    # =========================
    local_compute_time = t1 - t0
    total_compute_time = comm.reduce(local_compute_time, op=MPI.MAX, root=0)

    if rank == 0 and stats:

        print("\n" + "="*50)
        print("MPI KERNEL BENCHMARK")
        print("="*50)

        print(f"Wall Time (parallel): {total_compute_time:.4f} s")

        print("\n--- Algorithm Dynamics ---")
        print(f"Total iterations: {stats['total_iterations']}")
        print(f"Effective iterations: {stats['effective_iterations']}")

        print("\n--- Initial LS ---")
        print(f"Eval: {stats['initial_improvement']['n_evals']}")
        print(f"Exec: {stats['initial_improvement']['n_execs']}")
        print(f"Avg steps: {stats['initial_improvement']['avg_steps']:.2f}")

        print("\n--- Main LS ---")
        print(f"Eval: {stats['improvement']['n_evals']}")
        print(f"Exec: {stats['improvement']['n_execs']}")
        print(f"Avg steps: {stats['improvement']['avg_steps']:.2f}")

        print("="*50)

if __name__ == "__main__":
    # Example usage: mpirun -n 4 python pruebas/kernel_benchmark.py
    n_problem = 200
    if len(sys.argv) > 1:
        n_problem = int(sys.argv[1])
    
    benchmark_kernel(n_problem)
