import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

from src.problemas import RepairKnapsack
from src.RefSetMethods import ReferenceSet
from src.LocalSearchMethods import (LocalSearch,
    BitFlipMutationNeighborhood,
    SingleBitFlipNeighborhood,
    KnapsackNeighborhood,
    SingleBitActivateNeighborhood,
    MultipleBitActivateNeighborhood
    )
from src.CombinationMethods import (
    PymooCrossoverCombination, 
    ExhaustiveSubsetGeneration, 
    TournamentSubsetGeneration2, 
    BinaryTournamentSubsetGeneration)
from src.population import FrequencyBinaryDiversification
from src.ParallelScatterSearch import ScatterSearch
from src.config import config


from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.pntx import TwoPointCrossover, SinglePointCrossover
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.crossover.hux import HUX
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
from pymoo.problems.single.knapsack import create_random_knapsack_problem

def get_stagnation_stats(all_fitness, all_evals=None, pop_size=100):
    stagnation_gens = []
    stagnation_evals = []
    
    for i, run in enumerate(all_fitness):
        # Encontramos el mejor fitness alcanzado en esta ejecución
        best_val = np.min(run) # Usamos min porque pymoo minimiza por defecto
        
        # Encontramos el primer índice donde aparece ese valor
        first_gen_reached = np.where(np.array(run) == best_val)[0][0]
        stagnation_gens.append(first_gen_reached)
        
        # Calculamos evaluaciones
        if all_evals is not None:
            # Para Scatter Search usamos los datos capturados
            stagnation_evals.append(all_evals[i][first_gen_reached])
        else:
            # Para GA usamos la fórmula (gen * pop_size)
            stagnation_evals.append(first_gen_reached * pop_size)
            
    return {
        "mean_gen": np.mean(stagnation_gens),
        "std_gen": np.std(stagnation_gens),
        "mean_evals": np.mean(stagnation_evals),
        "std_evals": np.std(stagnation_evals)
    }


def run_experiment(problem,n_runs=10,seeds=None):


    print(f"Running {n_runs} trials for each algorithm on NK Landscape (N={N}, K={K})...")

    # Custom Callback to capture best fitness AND diversity
    class DetailedCallback(Callback):
        def __init__(self) -> None:
            super().__init__()
            self.data["best"] = []
            self.data["added"] = []
            self.data["n_evals"] = []
            
        def notify(self, algorithm):
            # 1. Capture Best Fitness
            if algorithm.opt is not None and len(algorithm.opt) > 0:
                val = algorithm.opt.get("F")
                self.data["best"].append(float(np.atleast_1d(val).flatten()[0]))
            elif algorithm.pop is not None and len(algorithm.pop) > 0:
                val = algorithm.pop.get("F")
                self.data["best"].append(float(np.min(val)))
            else:
                if self.data["best"]:
                    self.data["best"].append(self.data["best"][-1])
                else:
                    self.data["best"].append(0.0) 
            

            
            # 3. Capture RefSet Additions
            if hasattr(algorithm, "n_added"):
                self.data["added"].append(algorithm.n_added)
            else:
                self.data["added"].append(0)

            # 4. Capture Evaluated Solutions per iteration
            if hasattr(algorithm, "evaluator"):
                self.data["n_evals"].append(algorithm.evaluator.n_eval)
            else:
                self.data["n_evals"].append(0)
            

    all_ss_fitness, all_ss_added, all_ss_evals = [], [], []
    all_ga_fitness = []
    
    ss_times = []
    ga_times = []

    probflip = 1 / (N)
    neighbor_sample = 10

    for i in range(n_runs):
        if seeds is None:
            seed = 42 + i
        else:
            seed = seeds[i]
        print(f"Trial {i+1}/{n_runs} (Seed {seed})...")   

        # Vamos a probar a ver si jalaaaa AAAAHH

        subset_gen = BinaryTournamentSubsetGeneration()
        neighborhood_generator = MultipleBitActivateNeighborhood(n_bits=2,sample=30)
        combination_method = PymooCrossoverCombination(SinglePointCrossover())
        diversification_method = BinaryRandomSampling()
        starting_improv = LocalSearch(neighborhood=BitFlipMutationNeighborhood(prob_var=probflip,n_neighbors=5),
                             max_steps=1, 
                             method="best_improvement")
        improvement_method = LocalSearch(neighborhood=neighborhood_generator,
                                  max_steps=3, 
                                  method="best_improvement")

        ss_algorithm = ScatterSearch(
            subset_generation_method=subset_gen,
            combination_method=combination_method,
            diversification_method=diversification_method,
            initial_improvement_method=starting_improv,
            improvement_method=improvement_method,
            ReferenceSet=ReferenceSet('Hamming'),
            repair=RepairKnapsack(),
            reference_set_size=10,
            solution_pool_size=100
        )

        ga_algorithm = GA(
            pop_size=100,
            sampling=BinaryRandomSampling(),
            crossover=SinglePointCrossover(),
            repair= RepairKnapsack(),
            mutation=BitflipMutation(prob_var=probflip),
            eliminate_duplicates=True)

        # Run Scatter Search
        start = time.time()
        res_ss = minimize(problem, ss_algorithm, ("n_eval", n_gen*100), seed=seed, callback=DetailedCallback(), verbose=False)
        ss_times.append(time.time() - start)
        all_ss_fitness.append(res_ss.algorithm.callback.data["best"])
        all_ss_added.append(res_ss.algorithm.callback.data["added"])
        all_ss_evals.append(res_ss.algorithm.callback.data["n_evals"])
        
        # Run GA
        start = time.time()
        res_ga = minimize(problem, ga_algorithm, ('n_gen', n_gen), seed=seed, callback=DetailedCallback(), verbose=False)
        ga_times.append(time.time() - start)
        all_ga_fitness.append(res_ga.algorithm.callback.data["best"])


    
    # === INTERPOLACIÓN POR EVALUACIONES ===
    max_evals = n_gen * 100
    evals_grid = np.linspace(0, max_evals, 200)

    def interpolate_trial(evals, values, grid):
        # np.interp expects purely increasing x-coordinates.
        e = np.array(evals, dtype=float)
        v = np.array(values, dtype=float)
        
        # Resolve duplicates by adding a tiny epsilon
        for j in range(1, len(e)):
            if e[j] <= e[j-1]:
                e[j] = e[j-1] + 1e-6
                
        return np.interp(grid, e, v)

    # Procesar SS
    ss_f_interp = []
    ss_a_interp = []

    for i in range(n_runs):
        e_ss = all_ss_evals[i]
        # Pad with 0 at the start if needed
        if e_ss[0] > 0:
            e_ss = [0] + e_ss
            f_ss = [all_ss_fitness[i][0]] + all_ss_fitness[i]
            a_ss = [0] + all_ss_added[i]
        else:
            f_ss = all_ss_fitness[i]
            a_ss = all_ss_added[i]
            
        ss_f_interp.append(interpolate_trial(e_ss, f_ss, evals_grid))
        ss_a_interp.append(interpolate_trial(e_ss, a_ss, evals_grid))

    ss_f_mean = np.mean(ss_f_interp, axis=0)
    ss_f_std = np.std(ss_f_interp, axis=0) * 1.96 / np.sqrt(n_runs)
    ss_a_mean = np.mean(ss_a_interp, axis=0)
    ss_a_std = np.std(ss_a_interp, axis=0) * 1.96 / np.sqrt(n_runs)

    # Procesar GA
    ga_f_interp = []

    for i in range(n_runs):
        e_ga = np.arange(1, len(all_ga_fitness[i]) + 1) * 100
        if e_ga[0] > 0:
            e_ga = np.insert(e_ga, 0, 0)
            f_ga = [all_ga_fitness[i][0]] + all_ga_fitness[i]
        else:
            f_ga = all_ga_fitness[i]
            
        ga_f_interp.append(interpolate_trial(e_ga, f_ga, evals_grid))

    ga_f_mean = np.mean(ga_f_interp, axis=0)
    ga_f_std = np.std(ga_f_interp, axis=0) * 1.96 / np.sqrt(n_runs)

    # Visualization
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 12))
    
    # 1. Convergence Plot
    ax1.plot(evals_grid[16:], ss_f_mean[16:], label=f'Scatter Search (Avg Time: {np.mean(ss_times):.2f}s)', color='#636EFA', linewidth=2)
    ax1.fill_between(evals_grid[16:], ss_f_mean[16:] - ss_f_std[16:], ss_f_mean[16:] + ss_f_std[16:], color='#636EFA', alpha=0.2)
    ax1.plot(evals_grid[16:], ga_f_mean[16:], label=f'Genetic Algorithm (Avg Time: {np.mean(ga_times):.2f}s)', color='#EF553B', linewidth=2)
    ax1.fill_between(evals_grid[16:], ga_f_mean[16:] - ga_f_std[16:], ga_f_mean[16:] + ga_f_std[16:], color='#EF553B', alpha=0.2)
    ax1.set_title(f"Convergence Comparison | N={N}, K={K}", fontsize=16, fontweight='bold')
    ax1.set_xlabel("Evaluaciones de la Función Objetivo", fontsize=14)
    ax1.set_ylabel(r"Best Fitness (Mean)", fontsize=14)
    ax1.legend(fontsize=11)


    # 3. Additions Plot
    ax3.plot(evals_grid, ss_a_mean, label='New solutions added to RefSet', color='#00CC96', linewidth=2)
    ax3.fill_between(evals_grid, ss_a_mean - ss_a_std, ss_a_mean + ss_a_std, color='#00CC96', alpha=0.2)
    ax3.set_title("Scatter Search Dynamics: RefSet Replacement Rate", fontsize=16, fontweight='bold')
    ax3.set_xlabel("Evaluaciones de la Función Objetivo", fontsize=14)
    ax3.set_ylabel("Count (New Solutions)", fontsize=14)
    ax3.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig('comparison_with_metrics_cleaned.png', dpi=120)
    print("Plot saved as comparison_with_metrics_cleaned.png")

    # Boxplot del fitness final
    final_ss = [run[-1] for run in all_ss_fitness]
    final_ga = [run[-1] for run in all_ga_fitness]

    plt.figure(figsize=(8,6))
    plt.boxplot([final_ss, final_ga], tick_labels=['Scatter Search', 'Genetic Algorithm'])
    plt.title("Distribución del Fitness Final")
    plt.ylabel("Fitness")
    plt.savefig("boxplot_fitness.png", dpi=120)

    # --- Cálculo de Estancamiento ---
    ss_stagnation = get_stagnation_stats(all_ss_fitness, all_ss_evals)
    ga_stagnation = get_stagnation_stats(all_ga_fitness, pop_size=100)

    # Calcular Evals totales mean/std real para SS
    ss_e_totales = [run[-1] for run in all_ss_evals]
    ss_e_mean_final = np.mean(ss_e_totales)
    ss_e_std_final = np.std(ss_e_totales)

    print("\n" + "="*30)
    print("MÉTRICAS DE ESTANCAMIENTO")
    print("="*30)
    print(f"Scatter Search:")
    print(f"  - Gen. promedio de estancamiento: {ss_stagnation['mean_gen']:.2f} ± {ss_stagnation['std_gen']:.2f}")
    print(f"  - Evals promedio de estancamiento: {ss_stagnation['mean_evals']:.2f} ± {ss_stagnation['std_evals']:.2f}")
    print(f"  - Mejor fitness alcanzado: {min(final_ss):.4f}")
    print(f"  - Peor fitness alcanzado: {max(final_ss):.4f}")
    print(f"  - Media fitness alcanzado: {np.mean(final_ss):.4f} ± {np.std(final_ss):.4f}")
    print(f"  - Evals totales: {ss_e_mean_final:.2f} ± {ss_e_std_final:.2f}")
    print(f"  - Tiempo total: {np.mean(ss_times):.2f} ± {np.std(ss_times):.2f}")
    
    print(f"\nGenetic Algorithm:")
    print(f"  - Gen. promedio de estancamiento: {ga_stagnation['mean_gen']:.2f} ± {ga_stagnation['std_gen']:.2f}")
    print(f"  - Evals promedio de estancamiento: {ga_stagnation['mean_evals']:.2f} ± {ga_stagnation['std_evals']:.2f}")
    print(f"  - Mejor fitness alcanzado: {min(final_ga):.4f}")
    print(f"  - Peor fitness alcanzado: {max(final_ga):.4f}")
    print(f"  - Media fitness alcanzado: {np.mean(final_ga):.4f} ± {np.std(final_ga):.4f}")
    print(f"  - Tiempo total: {np.mean(ga_times):.2f} ± {np.std(ga_times):.2f}")
    print("="*30)


if __name__ == "__main__":
    # Configuration from config.json or defaults
    N = config.get("problem", "nk_landscape", "N", default=100)
    K = config.get("problem", "nk_landscape", "K", default=6)
    n_gen = config.get("algorithm", "n_gen", default=100)
    n_runs = config.get("experiment", "n_runs", default=30)
    seed = config.get("problem", "seed", default=42)

   

    problem= create_random_knapsack_problem(N,seed=seed)
    print(np.mean(problem.C),np.std(problem.C))

    run_experiment(problem, n_runs=n_runs)