import os
import sys
import numpy as np
from scipy.stats import linregress
from pymoo.optimize import minimize
from time import perf_counter   # precisión mejorada

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from problemas import QUBO
from RefSetMethods import ReferenceSet
from LocalSearchMethods import LocalSearchFirstImprovement, LocalSearch_BitFlipMutation
from CombinationMethods import ExhaustiveSubsetGeneration, PymooCrossoverCombination
from population import FrequencyBinaryDiversification
from ParallelScatterSearch import ScatterSearch
from pymoo.operators.crossover.pntx import SinglePointCrossover


def create_qubo_problem(n, seed, min_q=-4, max_q=8, diag_q=-10):
    np.random.seed(seed)
    Q = np.random.randint(min_q, max_q, size=(n, n))
    Q = (Q + Q.T) // 2

    np.fill_diagonal(Q, diag_q)   
    return QUBO(Q)


def build_algorithm(n_problem, comm=None):
    probflip = 2 / n_problem
    neighbor_sample = n_problem // 10

    return ScatterSearch(
        subset_generation_method=ExhaustiveSubsetGeneration(),
        combination_method=PymooCrossoverCombination(SinglePointCrossover()),
        diversification_method=FrequencyBinaryDiversification(),
        initial_improvement_method=LocalSearch_BitFlipMutation(
            prob_var=probflip,
            n_neighbors=neighbor_sample,
            max_steps=1
        ),
        improvement_method=LocalSearchFirstImprovement(
            neighbor_sample=neighbor_sample,
            max_steps=3,
        ),
        ReferenceSet=ReferenceSet("Hamming"),
        reference_set_size=12,
        solution_pool_size=128,
        diversity_threshold=probflip * 3.0,
        comm=comm
    )


def estimate_complexity_exponent(
    sizes=(80, 100, 150, 200, 300),
    repeats=5,
    n_gen=50
):
    times_mean = []
    times_std = []

    for n in sizes:
        runtimes = []

        # Warm-up run (no se mide, solo carga librerías y estructuras)
        problem = create_qubo_problem(n, seed=999)
        algorithm = build_algorithm(n, comm=None)
        minimize(problem, algorithm, ("n_gen", 5), seed=0, verbose=False)

        # Repeticiones medidas
        for rep in range(repeats):
            problem = create_qubo_problem(n, seed=100 + rep)
            algorithm = build_algorithm(n, comm=None)

            start = perf_counter()
            minimize(
                problem,
                algorithm,
                ("n_gen", n_gen),
                seed=rep,
                verbose=False
            )
            runtimes.append(perf_counter() - start)

        mean_t = np.mean(runtimes)
        std_t = np.std(runtimes)
        times_mean.append(mean_t)
        times_std.append(std_t)

        print(f"n={n:4d}  time={mean_t:.6f}s ± {std_t:.6f}")

    log_n = np.log(sizes)
    log_t = np.log(times_mean)

    slope, intercept, r_value, _, _ = linregress(log_n, log_t)

    print("\n===== RESULTADO =====")
    print(f"Exponente empírico α = {slope:.4f}")
    print(f"R² = {r_value**2:.4f}")

    return slope, intercept, times_mean, times_std


if __name__ == "__main__":
    estimate_complexity_exponent()
