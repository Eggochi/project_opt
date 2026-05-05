"""
Benchmark Estadístico de Escalabilidad para Scatter Search QUBO

Uso:

Strong Scaling:
    mpirun -n 1 python escalabilidad.py --strong
    mpirun -n 2 python escalabilidad.py --strong
    mpirun -n 4 python escalabilidad.py --strong

Weak Scaling:
    mpirun -n 1 python escalabilidad.py --weak
    mpirun -n 2 python escalabilidad.py --weak
    mpirun -n 4 python escalabilidad.py --weak
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from mpi4py import MPI
from pymoo.optimize import minimize

# Añadir src al path para que encuentre los módulos locales
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from problemas import QUBO, NKLandscape, RepairKnapsack
from RefSetMethods import ReferenceSet
from LocalSearchMethods import (LocalSearch, 
                               MultipleBitActivateNeighborhood, 
                               BitFlipMutationNeighborhood,
                               SingleBitFlipNeighborhood)
from CombinationMethods import (PymooCrossoverCombination, 
BinaryTournamentSubsetGeneration)
from ParallelScatterSearch import ScatterSearch, ScatterSearch2
from pymoo.operators.crossover.pntx import SinglePointCrossover
from pymoo.core.callback import Callback
from pymoo.problems.single.knapsack import create_random_knapsack_problem
from pymoo.operators.sampling.rnd import BinaryRandomSampling

# ============================================================
# UTILIDADES
# ============================================================

def summarize(values):
    return {
        "mean": np.mean(values),
        "std": np.std(values),
        "min": np.min(values),
        "max": np.max(values)
    }


def create_nk_landscape(n):
    k = 6
    return NKLandscape(N=n, K=k)


def build_algorithm(n_problem, n_evals, comm):
    probflip = 1/n_problem
    neighbor_sample = 5
    max_steps = 3
    diversity_threshold = 1/n_problem
    # Knapsack
    neighborhood = MultipleBitActivateNeighborhood(n_bits=2,sample=20)
    # QUBO or NKLandscape
    #neighborhood = SingleBitFlipNeighborhood(sample=5)
    
    return ScatterSearch(
        subset_generation_method=BinaryTournamentSubsetGeneration(),
        combination_method=PymooCrossoverCombination(SinglePointCrossover()),
        diversification_method=BinaryRandomSampling(),
        initial_improvement_method=LocalSearch(neighborhood=BitFlipMutationNeighborhood(prob_var=probflip,sample=10),
                             max_steps=1, 
                             method="best_improvement"),
        improvement_method=LocalSearch(neighborhood=neighborhood,
                                  max_steps=3, 
                                  method="best_improvement"),
        ReferenceSet=ReferenceSet("Hamming"),
        repair=RepairKnapsack(),
        reference_set_size=16,
        solution_pool_size=int(2**7),
        diversity_threshold=diversity_threshold,
        max_evals=n_evals,
        comm=comm
    )

# ============================================================
# STRONG SCALING
# ============================================================

def run_strong_scaling_benchmark(
    n_problem,
    n_evals,
    repeats=5,
    output_dir="pruebas"
):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    results_file = os.path.join(output_dir, "strong_scaling_stats.csv")

    #problem = create_nk_landscape(n_problem)
    problem = create_random_knapsack_problem(n_problem, seed=42)

    times = []

    for rep in range(repeats):

        algorithm = build_algorithm(n_problem, n_evals, comm)

        comm.barrier()
        if rank == 0:
            start = time.time()

        minimize(
            problem,
            algorithm,
            ("n_eval", n_evals),
            seed=42 + rep,
            verbose=False
        )

        comm.barrier()

        if rank == 0:
            times.append(time.time() - start)

    if rank != 0:
        return

    stats = summarize(times)

    os.makedirs(output_dir, exist_ok=True)

    existing = []

    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            next(f)
            for line in f:
                p, n, mean, std, mn, mx, sp, eff = line.strip().split(",")
                existing.append({
                    "n_procs": int(p),
                    "n_problem": int(n),
                    "mean": float(mean),
                    "std": float(std),
                    "min": float(mn),
                    "max": float(mx),
                    "speedup": float(sp),
                    "efficiency": float(eff)
                })

    existing = [r for r in existing if r["n_procs"] != size]

    existing.append({
        "n_procs": size,
        "n_problem": n_problem,
        **stats,
        "speedup": 0.0,
        "efficiency": 0.0
    })

    existing.sort(key=lambda x: x["n_procs"])

    baseline = next((r for r in existing if r["n_procs"] == 1), None)

    if baseline:
        t1 = baseline["mean"]

        for r in existing:
            r["speedup"] = t1 / r["mean"]
            r["efficiency"] = 100 * r["speedup"] / r["n_procs"]

    with open(results_file, "w") as f:
        f.write("n_procs,n_problem,mean,std,min,max,speedup,efficiency\n")
        for r in existing:
            f.write(
                f"{r['n_procs']},{r['n_problem']},"
                f"{r['mean']:.6f},{r['std']:.6f},"
                f"{r['min']:.6f},{r['max']:.6f},"
                f"{r['speedup']:.6f},{r['efficiency']:.2f}\n"
            )


# ============================================================
# WEAK SCALING
# ============================================================

def run_weak_scaling_benchmark(
    base_n,
    n_evals,
    repeats=5,
    output_dir="pruebas"
):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    n_problem = base_n*size

    results_file = os.path.join(output_dir, "weak_scaling_stats.csv")

    #problem = create_nk_landscape(n_problem)
    problem = create_random_knapsack_problem(n_problem, seed=42)

    times = []

    for rep in range(repeats):

        algorithm = build_algorithm(n_problem, n_evals, comm)

        comm.barrier()
        if rank == 0:
            start = time.time()

        minimize(
            problem,
            algorithm,
            ("n_eval", n_evals),
            seed=42 + rep,
            verbose=False
        )

        comm.barrier()

        if rank == 0:
            times.append(time.time() - start)

    if rank != 0:
        return

    stats = summarize(times)

    os.makedirs(output_dir, exist_ok=True)

    existing = []

    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            next(f)
            for line in f:
                p, n, mean, std, mn, mx, we = line.strip().split(",")
                existing.append({
                    "n_procs": int(p),
                    "n_problem": int(n),
                    "mean": float(mean),
                    "std": float(std),
                    "min": float(mn),
                    "max": float(mx),
                    "weak_efficiency": float(we)
                })

    existing = [r for r in existing if r["n_procs"] != size]

    existing.append({
        "n_procs": size,
        "n_problem": n_problem,
        **stats,
        "weak_efficiency": 0.0
    })

    existing.sort(key=lambda x: x["n_procs"])

    baseline = next((r for r in existing if r["n_procs"] == 1), None)

    if baseline:
        t1 = baseline["mean"]

        for r in existing:
            r["weak_efficiency"] = t1 / r["mean"]

    with open(results_file, "w") as f:
        f.write("n_procs,n_problem,mean,std,min,max,weak_efficiency\n")
        for r in existing:
            f.write(
                f"{r['n_procs']},{r['n_problem']},"
                f"{r['mean']:.6f},{r['std']:.6f},"
                f"{r['min']:.6f},{r['max']:.6f},"
                f"{r['weak_efficiency']:.6f}\n"
            )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--strong", action="store_true")
    parser.add_argument("--weak", action="store_true")

    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--base-n", type=int, default=50)

    parser.add_argument("--evals", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)

    args = parser.parse_args()

    rank = MPI.COMM_WORLD.Get_rank()

    if rank == 0:
        print("=== Benchmark Escalabilidad Scatter Search ===")

    if args.strong:
        run_strong_scaling_benchmark(
            n_problem=args.n,
            n_evals=args.evals,
            repeats=args.repeats
        )

    elif args.weak:
        run_weak_scaling_benchmark(
            base_n=args.base_n,
            n_evals=args.evals,
            repeats=args.repeats
        )

    else:
        if rank == 0:
            print("Debe especificar --strong o --weak")