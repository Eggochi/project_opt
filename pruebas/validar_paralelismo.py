import os
import sys
import time
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

class EvalCallback(Callback):
    def __init__(self, comm=None) -> None:
        super().__init__()
        self.comm = comm
        self.n_evals = 0
        self.history = []

        # Mejor fitness global observado hasta ahora
        self.best_so_far = None

    def notify(self, algorithm):

        local_evals = 0
        if hasattr(algorithm, 'n_evals'):
            local_evals = algorithm.n_evals
        elif hasattr(algorithm, 'evaluator') and hasattr(algorithm.evaluator, 'n_eval'):
            local_evals = algorithm.evaluator.n_eval

        # ==============================
        # Obtener evaluaciones GLOBALES (seguro ahora que has_next está sincronizado)
        # ==============================
        if self.comm is not None:
            self.n_evals = self.comm.allreduce(local_evals, op=MPI.SUM)
        else:
            self.n_evals = local_evals

        # ==============================
        # Obtener fitness actual (solo rank 0 tiene el opt actualizado)
        # ==============================
        current_f = None

        rank = self.comm.Get_rank() if self.comm is not None else 0

        if rank == 0 and algorithm.opt is not None and len(algorithm.opt) > 0:
            current_f = float(
                np.atleast_1d(
                    algorithm.opt.get("F")[0]
                ).flatten()[0]
            )

        # ==============================
        # Actualizar BEST-SO-FAR y guardar historia (solo rank 0)
        # ==============================
        if rank == 0:
            if current_f is not None:
                if self.best_so_far is None:
                    self.best_so_far = current_f
                else:
                    self.best_so_far = min(self.best_so_far, current_f)

            if self.best_so_far is not None:
                self.history.append(
                    (self.n_evals, self.best_so_far)
                )

def create_qubo_problem(n, seed=42):
    np.random.seed(seed)
    # Generar matriz Q simétrica
    Q = np.random.randint(-10, 10, size=(n, n))
    Q = (Q + Q.T) // 2
    return QUBO(Q)

def create_nk_problem(n, k):
    return NKLandscape(N=n, K=k)
    

def build_alg(comm, n_problem, max_evals=None):
    """
    Construye la configuración del algoritmo.
    Si comm es None, corre en modo secuencial.
    """
    probflip = 1/n_problem
    neighbor_sample = 5
    max_steps = 3
    diversity_threshold = 1/n_problem
    # Knapsack
    #neighborhood = MultipleBitActivateNeighborhood(n_bits=2,sample=20)
    # QUBO or NKLandscape
    neighborhood = SingleBitFlipNeighborhood(sample=5)
    
    return ScatterSearch(
        subset_generation_method=BinaryTournamentSubsetGeneration(),
        combination_method=PymooCrossoverCombination(SinglePointCrossover()),
        diversification_method=BinaryRandomSampling(),
        initial_improvement_method=LocalSearch(neighborhood=BitFlipMutationNeighborhood(prob_var=probflip,sample=10),
                             max_steps=1, 
                             method="best_improvement"),
        improvement_method=LocalSearch(neighborhood=neighborhood,
                                  max_steps=3, 
                                  method="first_improvement"),
        ReferenceSet=ReferenceSet("Hamming"),
        #repair=RepairKnapsack(),
        reference_set_size=16,
        solution_pool_size=int(2**7),
        diversity_threshold=diversity_threshold,
        max_evals=max_evals,
        comm=comm
    )

def run_comparison(n_problem=150, n_evals=5000, n_trials=10):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    #problem = create_random_knapsack_problem(n_problem, seed=42)
    #problem = create_qubo_problem(n_problem, seed=42)
    problem = create_nk_problem(n_problem, 5)

    # Listas para almacenar resultados de múltiples pruebas
    par_results = []

    if rank == 0:
        print(f"\n" + "="*60)
        print(f"PRUEBA DE RENDIMIENTO PARALELO (N={n_problem}, Evals={n_evals}, Trials={n_trials})")
        print("="*60)

    for trial in range(n_trials):
        current_seed = 42 + trial
        
        # --- 2. EJECUCIÓN PARALELA ---
        comm.Barrier()
        if rank == 0:
            print(f"[Trial {trial+1}/{n_trials}] Corriendo versión PARALELA con {size} procesos...")
        
        alg_par = build_alg(comm, n_problem, max_evals=n_evals)
        callback_par = EvalCallback(comm=comm)
        
        t0_par = MPI.Wtime()
        res_par = minimize(problem, alg_par, ("n_eval", 1000000), seed=current_seed, callback=callback_par, verbose=False)
        t1_par = MPI.Wtime()
        comm.Barrier() # Sincronizar fin: esperar a que TODOS los procesos terminen
        
        time_par = t1_par - t0_par
        # La callback ya contiene la suma global en n_evals debido al allreduce en notify()
        evals_par = callback_par.n_evals
        
        if rank == 0:
            best_f = res_par.F[0] if res_par.F is not None else float('inf')
            par_results.append({
                'fitness': best_f,
                'time': time_par,
                'evals': evals_par,
                'history': callback_par.history # Guardar historia del trial
            })
            print(f"   -> Par: Fitness: {best_f:.4f} | Tiempo: {time_par:.4f}s | Evals: {evals_par}")

    # --- 3. REPORTE FINAL ---
    if rank == 0:
        print("\n" + "="*60)
        print("REPORTE RESUMEN PARALELO (Mean ± Std)")
        print("="*60)
        
        metrics = ['fitness', 'time', 'evals']
        stats = {}
        
        for m in metrics:
            p_vals = [r[m] for r in par_results]
            stats[m] = {
                'mean': np.mean(p_vals),
                'std': np.std(p_vals)
            }
        
        print(f"{'Métrica':<15} | {'Paralela (avg ± std)':<25}")
        print("-" * 60)
        
        print(f"{'Fitness':<15} | {stats['fitness']['mean']:>9.6f} ± {stats['fitness']['std']:<10.6f}")
        print(f"{'Tiempo (s)':<15} | {stats['time']['mean']:>9.4f} ± {stats['time']['std']:<10.4f}")
        print(f"{'Evaluaciones':<15} | {stats['evals']['mean']:>9.0f} ± {stats['evals']['std']:<10.0f}")
        print("="*60 + "\n")

        # --- GUARDAR EN CSV ---
        csv_file = "parallel_performance_results.csv"
        file_exists = os.path.isfile(csv_file)
        
        row = {
            'n_procs': size,
            'n_problem': n_problem,
            'n_trials': n_trials,
            'f_mean': stats['fitness']['mean'],
            'f_std': stats['fitness']['std'],
            't_mean': stats['time']['mean'],
            't_std': stats['time']['std'],
            'e_mean': stats['evals']['mean'],
            'e_std': stats['evals']['std']
        }
        
        df_new = pd.DataFrame([row])
        df_new.to_csv(csv_file, mode='a', index=False, header=not file_exists)
        print(f"Resultados guardados en {csv_file}")

        # --- GUARDAR HISTORIA DETALLADA ---
        history_file = "parallel_convergence_history.csv"
        h_exists = os.path.isfile(history_file)
        
        h_rows = []
        for trial_idx, result in enumerate(par_results):
            for it_idx, (ev, fit) in enumerate(result['history']):
                h_rows.append({
                    'n_procs': size,
                    'trial': trial_idx,
                    'iteration': it_idx,
                    'evals': ev,
                    'fitness': fit
                })
        
        df_history = pd.DataFrame(h_rows)
        df_history.to_csv(history_file, mode='a', index=False, header=not h_exists)
        print(f"Historia de convergencia guardada en {history_file}")

if __name__ == "__main__":
    n_problem = 100
    n_evals = 1000
    n_trials = 30
    if len(sys.argv) > 1:
        n_problem = int(sys.argv[1])
    if len(sys.argv) > 2:
        n_evals = int(sys.argv[2])
    if len(sys.argv) > 3:
        n_trials = int(sys.argv[3])
        
    run_comparison(n_problem=n_problem, n_evals=n_evals, n_trials=n_trials)
