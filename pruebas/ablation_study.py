"""
Ablation Study - Scatter Search
================================
Prueba combinaciones de métodos en QUBO y Knapsack (1000 evals, SPC crossover,
BinaryRandomSampling diversification). Guarda los resultados en JSON y CSV,
e imprime el Top-10 de combinaciones por fitness medio.
"""
import os
import sys
import json
import itertools
import numpy as np
import pandas as pd
from mpi4py import MPI
from pymoo.optimize import minimize
from pymoo.operators.crossover.pntx import SinglePointCrossover
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.problems.single.knapsack import create_random_knapsack_problem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from problemas import QUBO, RepairKnapsack, RepairKnapsack2
from RefSetMethods import ReferenceSet
from LocalSearchMethods import (LocalSearch,
                                 BitFlipMutationNeighborhood,
                                 SingleBitFlipNeighborhood,
                                 KnapsackNeighborhood,
                                 SingleBitActivateNeighborhood,
                                 MultipleBitActivateNeighborhood)
from CombinationMethods import (
    ExhaustiveSubsetGeneration,
    BinaryTournamentSubsetGeneration,
    PymooCrossoverCombination,
)
from population import FrequencyBinaryDiversification
from ParallelScatterSearch import ScatterSearch

# ─────────────────────────────────────────────────────────────
# Problemas
# ─────────────────────────────────────────────────────────────
N_QUBO   = 50
N_KNAP   = 200
N_EVALS  = 5000
N_TRIALS = 10
SEED_BASE = 42

def make_qubo(seed=42):
    rng = np.random.default_rng(seed)
    Q = rng.integers(-10, 10, size=(N_QUBO, N_QUBO))
    Q = (Q + Q.T) // 2
    return QUBO(Q)

def make_knapsack(seed=42):
    return create_random_knapsack_problem(N_KNAP, seed=seed)

# ─────────────────────────────────────────────────────────────
# Espacio de búsqueda de métodos
# ─────────────────────────────────────────────────────────────
SUBSET_GENERATORS = {
    "BinaryTournament":  lambda: BinaryTournamentSubsetGeneration(),
    "Exhaustive_filt":   lambda: ExhaustiveSubsetGeneration(distance_threshold=3),
}

NEIGHBORHOOD_METHODS = {
    "SingleBitActivate":  lambda n, mode: SingleBitActivateNeighborhood(sample=20),
    "MultipleBitActivate":  lambda n, mode: MultipleBitActivateNeighborhood(n_bits=2,sample=20),
}

REFSET_SIZES = {
    "RefSet_8":  10,
    "RefSet_16": 15,
}

LOCAL_SEARCH_TYPES = {
    "FirstImprov":  "first_improvement",
    "BestImprov": "best_improvement",
}

REPAIR_METHODS = {
    "RepairKnapsack":  RepairKnapsack,
    "RepairKnapsack2":  RepairKnapsack2,
}

# Combinación crossover fija: SinglePointCrossover
CROSSOVER = SinglePointCrossover()

# ─────────────────────────────────────────────────────────────
# Construcción del algoritmo
# ─────────────────────────────────────────────────────────────
def build_alg(subset_key, refset_key, repair_key, n_key, ls_key, problem):
    comm = MPI.COMM_WORLD
    n = problem.n_var
    probflip = 1 / n

    return ScatterSearch(
        subset_generation_method=SUBSET_GENERATORS[subset_key](),
        combination_method=PymooCrossoverCombination(CROSSOVER),
        diversification_method=FrequencyBinaryDiversification(),
        initial_improvement_method=LocalSearch(
            neighborhood=BitFlipMutationNeighborhood(prob_var=probflip, n_neighbors=5),
            max_steps=1, metric_prefix="best_improvement"
        ),
        improvement_method=LocalSearch(neighborhood=NEIGHBORHOOD_METHODS[n_key](n, None),
        method=LOCAL_SEARCH_TYPES[ls_key],
        max_steps=3
        ),
        ReferenceSet=ReferenceSet("Hamming"),
        repair=REPAIR_METHODS[repair_key](),
        reference_set_size=REFSET_SIZES[refset_key],
        solution_pool_size=REFSET_SIZES[refset_key]*10,
        diversity_threshold=1 / n,
        comm=comm,
    )

# ─────────────────────────────────────────────────────────────
# Evaluación de una combinación
# ─────────────────────────────────────────────────────────────
def evaluate_combo(subset_key, refset_key, repair_key, n_key, ls_key, problem, label=""):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    n_evals_local = max(1, N_EVALS // size)

    fitness_results = []
    for trial in range(N_TRIALS):
        seed = SEED_BASE + trial
        alg = build_alg(subset_key, refset_key, repair_key, n_key, ls_key, problem)
        try:
            comm.Barrier()
            res = minimize(problem, alg, ("n_eval", n_evals_local),
                           seed=seed, verbose=False)
            comm.Barrier()
            local_f = float(res.F[0]) if rank == 0 else 0.0
            all_f = comm.gather(local_f, root=0)
            if rank == 0:
                fitness_results.append(local_f)
        except Exception as e:
            comm.Barrier()
            if rank == 0:
                print(f"  [ERROR] {label} trial {trial}: {e}")
                fitness_results.append(float("inf"))

    if rank == 0:
        return {
            "subset": subset_key,
            "refset": refset_key,
            "repair": repair_key,
            "neighborhood": n_key,
            "ls_type": ls_key,
            "problem": label,
            "mean": float(np.mean(fitness_results)),
            "std":  float(np.std(fitness_results)),
            "min":  float(np.min(fitness_results)),
            "trials": fitness_results,
        }
    return None

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    problems = {
        #"QUBO":     (make_qubo(),     None),
        "Knapsack": (make_knapsack(), RepairKnapsack()),
    }

    combos = list(itertools.product(
        SUBSET_GENERATORS.keys(),
        REFSET_SIZES.keys(),
        REPAIR_METHODS.keys(),
        NEIGHBORHOOD_METHODS.keys(),
        LOCAL_SEARCH_TYPES.keys()
    ))

    total = len(combos) * len(problems)
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"ABLATION STUDY  –  {total} combinaciones × {N_TRIALS} trials")
        print(f"Problemas: {list(problems.keys())}")
        print(f"{'='*60}\n")

    all_results = []
    idx = 0
    for prob_name, (problem, repair) in problems.items():
        for (sub, ref, repair_k, n_k, ls_k) in combos:
            idx += 1
            label = f"{sub}|{ref}|{repair_k}|{n_k}|{ls_k}|{prob_name}"
            if rank == 0:
                print(f"[{idx}/{total}] {label} ...", flush=True)

            row = evaluate_combo(sub, ref, repair_k, n_k, ls_k, problem, label=prob_name)
            if rank == 0 and row is not None:
                all_results.append(row)

    if rank == 0:
        # ── Guardar resultados completos ──────────────────────────
        out_dir = os.path.join(os.path.dirname(__file__), "..")
        json_path = os.path.join(out_dir, "ablation_results.json")
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)

        # ── Construir DataFrame ───────────────────────────────────
        df = pd.DataFrame([{k: v for k, v in r.items() if k != "trials"} for r in all_results])
        csv_path = os.path.join(out_dir, "ablation_results.csv")
        df.to_csv(csv_path, index=False)

        # ── Top-5 por problema → Top-10 total ────────────────────
        TOP_PER_PROBLEM = 5
        top10_rows = []
        df_top_parts = []

        for prob_name in df["problem"].unique():
            df_prob = df[df["problem"] == prob_name].sort_values("mean").head(TOP_PER_PROBLEM)
            df_top_parts.append(df_prob)
            for _, row in df_prob.iterrows():
                orig = next((r for r in all_results
                             if r["subset"] == row["subset"]
                             and r["refset"] == row["refset"]
                             and r["repair"] == row["repair"]
                             and r["neighborhood"] == row["neighborhood"]
                             and r["ls_type"] == row["ls_type"]
                             and r["problem"] == row["problem"]), None)
                if orig:
                    top10_rows.append(orig)

        df_top = pd.concat(df_top_parts)

        print(f"\n{'='*60}")
        print(f"TOP-{TOP_PER_PROBLEM} POR PROBLEMA (mejor fitness medio)")
        print(f"{'='*60}")
        for prob_name in df["problem"].unique():
            print(f"\n  >> {prob_name}:")
            sub_df = df_top[df_top["problem"] == prob_name]
            print(sub_df[["subset", "refset", "repair", "neighborhood", "ls_type", "mean", "std", "min"]].to_string(index=False))

        # Guardar top-10
        top10_path = os.path.join(out_dir, "ablation_top10.json")
        with open(top10_path, "w") as f:
            json.dump(top10_rows, f, indent=2)

        print(f"\nResultados guardados en:")
        print(f"  {json_path}")
        print(f"  {csv_path}")
        print(f"  {top10_path}")

if __name__ == "__main__":
    main()
