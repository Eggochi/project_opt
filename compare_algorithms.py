import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

from problemas import NKLandscape, QUBO
from RefSetMethods import ReferenceSet
from LocalSearchMethods import LocalSearchImprovement, LocalSearch_BitFlipMutation
from CombinationMethods import PymooCrossoverCombination, ExhaustiveSubsetGeneration, TournamentSubsetGeneration2, BinaryTournamentSubsetGeneration
from population import FrequencyBinaryDiversification, FrequencyBinaryDiversification2
from ParallelScatterSearch import ScatterSearch


from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.pntx import TwoPointCrossover, SinglePointCrossover
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.optimize import minimize
from pymoo.core.callback import Callback

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


def run_experiment(problem,n_runs=5,seeds=None):


    print(f"Running {n_runs} trials for each algorithm on NK Landscape (N={N}, K={K})...")

    # Custom Callback to capture best fitness AND diversity
    class DetailedCallback(Callback):
        def __init__(self) -> None:
            super().__init__()
            self.data["best"] = []
            self.data["diversity"] = []
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
            
            # 2. Capture Diversity
            if algorithm.pop is not None and len(algorithm.pop) > 1:
                X = algorithm.pop.get("X")
                dists = pdist(np.atleast_2d(X), metric='hamming')
                self.data["diversity"].append(np.mean(dists) * 100.0)
            else:
                self.data["diversity"].append(0.0)
            
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
            

    all_ss_fitness, all_ss_diversity, all_ss_added, all_ss_evals = [], [], [], []
    all_ga_fitness, all_ga_diversity, all_ga_added = [], [], []
    ss_times, ga_times = [], []

    for i in range(n_runs):
        if seeds is None:
            seed = 42 + i
        else:
            seed = seeds[i]
        print(f"Trial {i+1}/{n_runs} (Seed {seed})...")
        
        probflip = 1/N

        # Vamos a probar a ver si jalaaaa AAAAHH
        combination_method = TwoPointCrossover()
        diversification_method = BinaryRandomSampling()
        starting_improv = LocalSearch_BitFlipMutation(prob_var=probflip, n_neighbors=2, max_steps=2)
        improvement_method = LocalSearch_BitFlipMutation(prob_var=probflip, n_neighbors=3, max_steps=3)
    
        # Initialize the algorithm
        ss_algorithm = ScatterSearch(
            subset_generation_method=BinaryTournamentSubsetGeneration(),
            combination_method=PymooCrossoverCombination(combination_method),
            diversification_method=diversification_method,
            initial_improvement_method=starting_improv,
            improvement_method=improvement_method,
            ReferenceSet=ReferenceSet('Hamming'),
            reference_set_size=10,
            solution_pool_size=100,
            diversity_threshold=probflip*2
        )

        ga_algorithm = GA(
            pop_size=100,
            sampling=BinaryRandomSampling(),
            crossover=combination_method,
            mutation=BitflipMutation(prob_var=probflip),
            eliminate_duplicates=True)

        # Run Scatter Search
        start = time.time()
        res_ss = minimize(problem, ss_algorithm, ('n_gen', n_gen), seed=seed, callback=DetailedCallback(), verbose=False)
        ss_times.append(time.time() - start)
        all_ss_fitness.append(res_ss.algorithm.callback.data["best"])
        all_ss_diversity.append(res_ss.algorithm.callback.data["diversity"])
        all_ss_added.append(res_ss.algorithm.callback.data["added"])
        all_ss_evals.append(res_ss.algorithm.callback.data["n_evals"])
        
        # Run GA
        start = time.time()
        res_ga = minimize(problem, ga_algorithm, ('n_gen', n_gen), seed=seed, callback=DetailedCallback(), verbose=False)
        ga_times.append(time.time() - start)
        all_ga_fitness.append(res_ga.algorithm.callback.data["best"])
        all_ga_diversity.append(res_ga.algorithm.callback.data["diversity"])
        all_ga_added.append([0] * len(res_ga.algorithm.callback.data["best"]))


    
    # ARREGLO CLAVE: Quitar el primer elemento [1:] para evitar el "punto cero"
    ss_f_mean = np.mean(all_ss_fitness, axis=0)[1:]
    ss_f_std = np.std(all_ss_fitness, axis=0)[1:]
    ga_f_mean = np.mean(all_ga_fitness, axis=0)[1:]
    ga_f_std = np.std(all_ga_fitness, axis=0)[1:]
    
    ss_d_mean = np.mean(all_ss_diversity, axis=0)[1:]
    ss_d_std = np.std(all_ss_diversity, axis=0)[1:]
    ga_d_mean = np.mean(all_ga_diversity, axis=0)[1:]
    ga_d_std = np.std(all_ga_diversity, axis=0)[1:]
    
    ss_a_mean = np.mean(all_ss_added, axis=0)[1:]
    ss_a_std = np.std(all_ss_added, axis=0)[1:]

    ss_e_mean = np.mean(all_ss_evals, axis=0)
    ss_e_std = np.std(all_ss_evals, axis=0)
    ga_evals = 100 * np.arange(1, n_gen)

    # Visualization
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 16))
    
    # Nuevo eje X ajustado (empezando en generación 1)
    gens = np.arange(1, n_gen)
    
    # 1. Convergence Plot
    # Calbiar la std a intervalo de confianza del 95%
    ss_f_std = ss_f_std * 1.96 / np.sqrt(n_runs)
    ga_f_std = ga_f_std * 1.96 / np.sqrt(n_runs)
    ax1.plot(gens, ss_f_mean, label=f'Scatter Search (Avg Time: {np.mean(ss_times):.2f}s)', color='#636EFA', linewidth=2)
    ax1.fill_between(gens, ss_f_mean - ss_f_std, ss_f_mean + ss_f_std, color='#636EFA', alpha=0.2)
    ax1.plot(gens, ga_f_mean, label=f'Genetic Algorithm (Avg Time: {np.mean(ga_times):.2f}s)', color='#EF553B', linewidth=2)
    ax1.fill_between(gens, ga_f_mean - ga_f_std, ga_f_mean + ga_f_std, color='#EF553B', alpha=0.2)
    ax1.set_title(f"Convergence Comparison | N={N}, K={K}", fontsize=16, fontweight='bold')
    ax1.set_ylabel(r"Best Fitness (Mean)", fontsize=14)
    ax1.legend(fontsize=11)

    # 2. Diversity Plot
    ax2.plot(gens, ss_d_mean, label='Scatter Search Diversity', color='#636EFA', linestyle='--', linewidth=2)
    ax2.fill_between(gens, ss_d_mean - ss_d_std, ss_d_mean + ss_d_std, color='#636EFA', alpha=0.1)
    ax2.plot(gens, ga_d_mean, label='Genetic Algorithm Diversity', color='#EF553B', linestyle='--', linewidth=2)
    ax2.fill_between(gens, ga_d_mean - ga_d_std, ga_d_mean + ga_d_std, color='#EF553B', alpha=0.1)
    ax2.set_title("Population Diversity (Hamming Distance %)", fontsize=16, fontweight='bold')
    ax2.set_ylabel("Diversity %", fontsize=14)
    ax2.legend(fontsize=11)

    # 3. Additions Plot
    ax3.plot(gens, ss_a_mean, label='New solutions added to RefSet', color='#00CC96', linewidth=2)
    ax3.fill_between(gens, ss_a_mean - ss_a_std, ss_a_mean + ss_a_std, color='#00CC96', alpha=0.2)
    ax3.set_title("Scatter Search Dynamics: RefSet Replacement Rate", fontsize=16, fontweight='bold')
    ax3.set_xlabel("Generation", fontsize=14)
    ax3.set_ylabel("Count (New Solutions)", fontsize=14)
    ax3.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig('comparison_with_metrics_cleaned.png', dpi=120)
    print("Plot saved as comparison_with_metrics_cleaned.png")

    # Boxplot del fitness final
    final_ss = [run[-1] for run in all_ss_fitness]
    final_ga = [run[-1] for run in all_ga_fitness]

    plt.figure(figsize=(8,6))
    plt.boxplot([final_ss, final_ga], labels=['Scatter Search', 'Genetic Algorithm'])
    plt.title("Distribución del Fitness Final")
    plt.ylabel("Fitness")
    plt.savefig("boxplot_fitness.png", dpi=120)

    # Evaluations per generation
    # Calbiar la std a intervalo de confianza del 95%
    ss_e_std = ss_e_std * 1.96 / np.sqrt(n_runs)
    gens = np.arange(1, n_gen+1)
    ga_evals = 100 * gens

    plt.figure(figsize=(8,6))
    plt.plot(gens, ss_e_mean, label='Scatter Search Evaluations', color='#636EFA', linewidth=2)
    plt.fill_between(gens, ss_e_mean - ss_e_std, ss_e_mean + ss_e_std, color='#636EFA', alpha=0.2)
    plt.plot(gens, ga_evals, label='Genetic Algorithm Evaluations', color='#EF553B', linewidth=2)
    plt.title("Evaluations per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Evaluations")
    plt.legend()
    plt.savefig("evaluations_per_generation.png", dpi=120)

    # --- Cálculo de Estancamiento ---
    ss_stagnation = get_stagnation_stats(all_ss_fitness, all_ss_evals)
    ga_stagnation = get_stagnation_stats(all_ga_fitness, pop_size=100)

    print("\n" + "="*30)
    print("MÉTRICAS DE ESTANCAMIENTO")
    print("="*30)
    print(f"Scatter Search:")
    print(f"  - Gen. promedio de estancamiento: {ss_stagnation['mean_gen']:.2f} ± {ss_stagnation['std_gen']:.2f}")
    print(f"  - Evals promedio de estancamiento: {ss_stagnation['mean_evals']:.2f} ± {ss_stagnation['std_evals']:.2f}")
    print(f"  - Mejor fitness alcanzado: {min(final_ss):.4f}")
    print(f"  - Peor fitness alcanzado: {max(final_ss):.4f}")
    print(f"  - Evals totales: {ss_e_mean[-1]:.2f} ± {ss_e_std[-1]:.2f}")
    print(f"  - Tiempo total: {np.mean(ss_times):.2f} ± {np.std(ss_times):.2f}")
    
    print(f"\nGenetic Algorithm:")
    print(f"  - Gen. promedio de estancamiento: {ga_stagnation['mean_gen']:.2f} ± {ga_stagnation['std_gen']:.2f}")
    print(f"  - Evals promedio de estancamiento: {ga_stagnation['mean_evals']:.2f} ± {ga_stagnation['std_evals']:.2f}")
    print(f"  - Mejor fitness alcanzado: {min(final_ga):.4f}")
    print(f"  - Peor fitness alcanzado: {max(final_ga):.4f}")
    print(f"  - Tiempo total: {np.mean(ga_times):.2f} ± {np.std(ga_times):.2f}")
    print("="*30)


if __name__ == "__main__":
    # Configuration
    N, K = 100, 6
    n_gen = 80
    problem = NKLandscape(N=N, K=K)


    #QUBO
    np.random.seed(42)
    Q = np.random.randint(-5, 10, size=(N, N))
    Q = (Q + Q.T) // 2
    #Turn odd index values into -10
    #for i in range(N):
    #    for j in range(i+1,N):
    #        if i % 2 == 0:
    #            Q[i, j] = -10
    #            Q[j, i] = -10
    problem2 = QUBO(Q)

    #run_experiment(problem,n_runs=30)
    run_experiment(problem2,n_runs=30)