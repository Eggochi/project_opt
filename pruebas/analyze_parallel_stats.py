import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

# Importar funciones desde el script original si es posible, o simplemente copiarlas
# Para que sea autocontenido y fácil de correr, copiaré las funciones clave.

def holm_bonferroni(p_values, alpha=0.05):
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p   = np.array(p_values)[sorted_idx]
    adjusted   = np.zeros(n)
    for rank, idx in enumerate(sorted_idx):
        adjusted[idx] = min(1.0, sorted_p[rank] * (n - rank))
    return adjusted

def run_analysis(data, output_img="parallel_statistical_analysis.png"):
    labels = [str(r["n_procs"]) + " Procs" for r in data]
    n_methods = len(data)

    # ── Matriz de fitness (filas=trials, cols=métodos) ────────
    # Asegurarnos de que todos tengan el mismo número de trials
    min_trials = min(len(r["trials"]) for r in data)
    trial_matrix = np.array([r["trials"][:min_trials] for r in data], dtype=float).T
    
    n_trials, n_m = trial_matrix.shape
    print(f"\nAnalizando {n_m} configuraciones con {n_trials} trials cada una.\n")

    # ── 1. Test de Friedman ───────────────────────────────────
    stat_f, p_friedman = stats.friedmanchisquare(*[trial_matrix[:, j] for j in range(n_m)])
    print(f"{'='*60}")
    print(f"TEST DE FRIEDMAN")
    print(f"{'='*60}")
    print(f"  Estadístico χ²: {stat_f:.4f}")
    print(f"  p-valor       : {p_friedman:.66f}")
    if p_friedman < 0.05:
        print("  ✅ Diferencias SIGNIFICATIVAS entre métodos (p < 0.05)")
    else:
        print("  ⚠️  Sin diferencias significativas detectadas (p ≥ 0.05)")

    # ── Rangos de Friedman ────────────────────────────────────
    ranks = np.zeros((n_trials, n_m))
    for i in range(n_trials):
        ranks[i] = stats.rankdata(trial_matrix[i])  # menor = mejor
    mean_ranks = ranks.mean(axis=0)

    rank_df = pd.DataFrame({
        "Config": labels,
        "Mean_Rank": mean_ranks,
        "Mean_Fitness": trial_matrix.mean(axis=0),
        "Std_Fitness": trial_matrix.std(axis=0),
    }).sort_values("Mean_Rank")

    print(f"\nRANKING DE FRIEDMAN (Menor Rango = Mejor):")
    print(rank_df.to_string(index=False))

    # ── 2. Post-hoc Wilcoxon ──────────────────────────────────
    print(f"\n{'='*60}")
    print(f"POST-HOC: WILCOXON (con corrección Holm)")
    print(f"{'='*60}")
    
    pairs = list(combinations(range(n_m), 2))
    p_values = []
    pair_labels = []

    for i, j in pairs:
        try:
            _, p = stats.wilcoxon(trial_matrix[:, i], trial_matrix[:, j])
            p_values.append(p)
        except ValueError: # Si todos los valores son idénticos
            p_values.append(1.0)
        pair_labels.append((i, j))

    adj_p = holm_bonferroni(p_values)

    results = []
    p_matrix = np.ones((n_m, n_m))
    
    for idx, (i, j) in enumerate(pair_labels):
        p_matrix[i, j] = adj_p[idx]
        p_matrix[j, i] = adj_p[idx]
        results.append({
            "A": labels[i],
            "B": labels[j],
            "p_raw": p_values[idx],
            "p_adj": adj_p[idx],
            "Signif": "SI" if adj_p[idx] < 0.05 else "no"
        })

    res_df = pd.DataFrame(results)
    print(res_df[res_df["Signif"] == "SI"].to_string(index=False) if any(res_df["Signif"]=="SI") else "No hay diferencias significativas post-hoc.")

    # ── 3. Visualización ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Heatmap
    ax = axes[0]
    mask = np.triu(np.ones_like(p_matrix, dtype=bool))
    sns.heatmap(p_matrix, annot=True, mask=mask, cmap="YlGnBu_r", 
                xticklabels=labels, yticklabels=labels, ax=ax, fmt=".3f", cbar_kws={'label': 'p-valor ajustado'})
    ax.set_title("P-valores Wilcoxon (Holm)\n< 0.05 = Dif. Significativa")

    # Friedman Ranks
    ax2 = axes[1]
    best_idx = np.argmin(mean_ranks)
    colors = ["#2ecc71" if i == best_idx else "#3498db" for i in range(n_m)]
    ax2.barh(labels, mean_ranks, color=colors)
    ax2.invert_yaxis()
    ax2.set_xlabel("Rango medio (Menor es mejor)")
    ax2.set_title("Ranking de Friedman")

    plt.tight_layout()
    plt.savefig(output_img, dpi=150)
    print(f"\nGráfico guardado en {output_img}")

def load_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    # Crear ID de trial único (cada vez que iteration es 0)
    df['unique_trial'] = (df['iteration'] == 0).cumsum()
    
    # Obtener el fitness final de cada trial
    final_fits = df.sort_values('evals').groupby(['n_procs', 'unique_trial']).tail(1)
    
    data = []
    for n_proc in sorted(df['n_procs'].unique()):
        trials_fitness = final_fits[final_fits['n_procs'] == n_proc]['fitness'].tolist()
        data.append({
            "n_procs": n_proc,
            "trials": trials_fitness
        })
    return data

if __name__ == "__main__":
    csv_path = "Graficas/Knapsack_paralelo_200/SS1/parallel_convergence_history.csv"
    if not os.path.exists(csv_path):
        print(f"Error: No se encuentra {csv_path}")
        sys.exit(1)
        
    data = load_from_csv(csv_path)
    run_analysis(data)
