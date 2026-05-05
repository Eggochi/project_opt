"""
Statistical Analysis - Friedman + Wilcoxon Post-hoc
====================================================
Carga el Top-10 del ablation study y ejecuta:
  1. Test de Friedman para detectar diferencias significativas
  2. Prueba de Wilcoxon (rangos con signo) para cada par del Top-10
  3. Corrección de Holm-Bonferroni sobre los p-valores
  4. Ranking final basado en Friedman + p-valores
  5. Genera tabla y heatmap de p-valores
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

# ─────────────────────────────────────────────────────────────
# Rutas
# ─────────────────────────────────────────────────────────────
BASE_DIR  = os.path.join(os.path.dirname(__file__), "..")
TOP10_PATH = os.path.join(BASE_DIR, "ablation_top10.json")

# ─────────────────────────────────────────────────────────────
# Carga de datos
# ─────────────────────────────────────────────────────────────
def load_top10(path):
    with open(path) as f:
        data = json.load(f)
    return data

# ─────────────────────────────────────────────────────────────
# Etiquetas cortas
# ─────────────────────────────────────────────────────────────
def short_label(row):
    sub_abbr = {
        "Exhaustive":       "Exh",
        "Tournament_k2":    "Tur",
        "Tournament2_k2":   "Tu2",
        "BinaryTournament": "Bin",
        "Exhaustive_filt":  "ExF",
    }
    imp_abbr = {
        "FirstImprov_s3": "FI3",
        "BitFlip_fi_s1":  "BF1",
        "BitFlip_fi_s3":  "BF3",
        "BitFlip_bi_s1":  "BB1",
    }
    ref_abbr = {"RefSet_8": "R8", "RefSet_16": "R16"}
    s = sub_abbr.get(row["subset"], row["subset"][:4])
    r = ref_abbr.get(row["refset"], row["refset"])
    p = row["problem"][:3]
    
    n_str = row.get("neighborhood", "")[:4]
    ls_str = row.get("ls_type", "")[:2]
    rep_str = row.get("repair", "")[-4:] if "repair" in row else ""
    return f"{s}-{n_str}{ls_str}-{rep_str}-{r}-{p}"

# ─────────────────────────────────────────────────────────────
# Corrección de Holm-Bonferroni
# ─────────────────────────────────────────────────────────────
def holm_bonferroni(p_values, alpha=0.05):
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p   = np.array(p_values)[sorted_idx]
    adjusted   = np.zeros(n)
    for rank, idx in enumerate(sorted_idx):
        adjusted[idx] = min(1.0, sorted_p[rank] * (n - rank))
    return adjusted

# ─────────────────────────────────────────────────────────────
# Análisis estadístico
# ─────────────────────────────────────────────────────────────
def run_analysis(data):
    labels = [short_label(r) for r in data]
    n_methods = len(data)

    # ── Matriz de fitness (filas=trials, cols=métodos) ────────
    trial_matrix = np.array([r["trials"] for r in data], dtype=float).T
    n_trials, n_m = trial_matrix.shape
    print(f"\nAnalizando {n_m} configuraciones con {n_trials} trials cada una.\n")

    # ── 1. Test de Friedman ───────────────────────────────────
    stat_f, p_friedman = stats.friedmanchisquare(*[trial_matrix[:, j] for j in range(n_m)])
    print(f"{'='*60}")
    print(f"TEST DE FRIEDMAN")
    print(f"{'='*60}")
    print(f"  Estadístico χ²: {stat_f:.4f}")
    print(f"  p-valor       : {p_friedman:.6f}")
    if p_friedman < 0.05:
        print("  ✅ Diferencias SIGNIFICATIVAS entre métodos (p < 0.05)")
    else:
        print("  ⚠️  Sin diferencias significativas detectadas (p ≥ 0.05)")

    # ── Rangos de Friedman ────────────────────────────────────
    ranks = np.zeros((n_trials, n_m))
    for i in range(n_trials):
        ranks[i] = stats.rankdata(trial_matrix[i])  # menor = mejor (minimización)
    mean_ranks = ranks.mean(axis=0)

    rank_df = pd.DataFrame({
        "Config": labels,
        "Mean_Rank": mean_ranks,
        "Mean_Fitness": trial_matrix.mean(axis=0),
        "Std_Fitness": trial_matrix.std(axis=0),
    }).sort_values("Mean_Rank")

    print(f"\n{'='*60}")
    print("RANKING DE FRIEDMAN (menor rango = mejor)")
    print(f"{'='*60}")
    print(rank_df.to_string(index=False))

    # ── 2. Post-hoc: Wilcoxon por pares ──────────────────────
    print(f"\n{'='*60}")
    print("POST-HOC: WILCOXON RANGOS CON SIGNO (todos los pares)")
    print(f"{'='*60}")

    pair_results = []
    raw_pvals    = []
    pair_labels  = []

    for i, j in combinations(range(n_m), 2):
        xi = trial_matrix[:, i]
        xj = trial_matrix[:, j]
        # Wilcoxon necesita que no todos los valores sean iguales
        if np.all(xi == xj):
            p_val = 1.0
            stat  = 0.0
        else:
            stat, p_val = stats.wilcoxon(xi, xj, alternative="two-sided", zero_method="wilcox")
        raw_pvals.append(p_val)
        pair_labels.append((labels[i], labels[j]))
        pair_results.append({
            "A":       labels[i],
            "B":       labels[j],
            "stat_W":  round(float(stat), 4),
            "p_raw":   round(float(p_val), 6),
        })

    # ── Corrección Holm-Bonferroni ────────────────────────────
    adj_pvals = holm_bonferroni(raw_pvals)
    for k, row in enumerate(pair_results):
        row["p_adj_holm"] = round(float(adj_pvals[k]), 6)
        row["significant"] = adj_pvals[k] < 0.05

    wilcoxon_df = pd.DataFrame(pair_results).sort_values("p_adj_holm")
    print(wilcoxon_df.to_string(index=False))

    # ── 3. Matriz de p-valores ajustados ──────────────────────
    pval_matrix = pd.DataFrame(np.ones((n_m, n_m)), index=labels, columns=labels)
    for k, (li, lj) in enumerate(pair_labels):
        pval_matrix.loc[li, lj] = adj_pvals[k]
        pval_matrix.loc[lj, li] = adj_pvals[k]

    # ── 4. Identificación del mejor método ───────────────────
    # Método con menor rango medio de Friedman que es significativamente mejor
    # que la mayor cantidad de competidores
    sig_wins = np.zeros(n_m)
    for k, (i, j) in enumerate(combinations(range(n_m), 2)):
        if adj_pvals[k] < 0.05:
            mi = trial_matrix[:, i].mean()
            mj = trial_matrix[:, j].mean()
            if mi < mj:
                sig_wins[i] += 1
            else:
                sig_wins[j] += 1

    rank_df["Sig_Wins"] = [sig_wins[list(rank_df.index).index(i)]
                            if i in range(n_m) else 0
                            for i in range(n_m)]
    # Reordenar sig_wins de acuerdo al orden de rank_df
    rank_df_reset = rank_df.reset_index(drop=True)
    rank_df_reset["Sig_Wins"] = sig_wins[np.argsort(mean_ranks)]

    print(f"\n{'='*60}")
    print("CONCLUSIÓN: MEJOR MÉTODO")
    print(f"{'='*60}")
    best_idx = int(np.argmin(mean_ranks))
    print(f"  🏆 Mejor configuración (menor rango Friedman): {labels[best_idx]}")
    print(f"     Rango medio   : {mean_ranks[best_idx]:.3f}")
    print(f"     Fitness medio : {trial_matrix[:, best_idx].mean():.4f} ± {trial_matrix[:, best_idx].std():.4f}")
    print(f"     Victorias sig.: {int(sig_wins[best_idx])}/{n_m - 1} comparaciones")

    # ── 5. Guardar CSV de resultados ──────────────────────────
    wilcoxon_df.to_csv(os.path.join(BASE_DIR, "wilcoxon_results.csv"), index=False)
    rank_df_reset.to_csv(os.path.join(BASE_DIR, "friedman_ranking.csv"), index=False)
    print(f"\nArchivos guardados:")
    print(f"  {os.path.join(BASE_DIR, 'wilcoxon_results.csv')}")
    print(f"  {os.path.join(BASE_DIR, 'friedman_ranking.csv')}")

    # ── 6. Heatmap de p-valores ───────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Heatmap p-valores (Wilcoxon ajustado)
    ax = axes[0]
    mask = np.eye(n_m, dtype=bool)
    pval_arr = pval_matrix.values.astype(float)
    sns.heatmap(
        pval_arr, annot=True, fmt=".3f",
        xticklabels=labels, yticklabels=labels,
        cmap="RdYlGn_r", vmin=0, vmax=0.2,
        mask=mask, ax=ax, linewidths=0.5,
        annot_kws={"size": 7}
    )
    ax.set_title("P-valores Wilcoxon (ajustados Holm)\n< 0.05 = diferencia significativa", fontsize=11)
    ax.tick_params(axis='x', rotation=45, labelsize=7)
    ax.tick_params(axis='y', rotation=0,  labelsize=7)

    # Gráfico de barras: rango medio Friedman
    ax2 = axes[1]
    colors = ["#2ecc71" if i == best_idx else "#3498db" for i in range(n_m)]
    bars = ax2.barh(labels, mean_ranks, color=colors, edgecolor="white")
    ax2.axvline(x=mean_ranks.mean(), color="red", linestyle="--", linewidth=1.2, label="Rango promedio")
    ax2.set_xlabel("Rango medio de Friedman (menor = mejor)")
    ax2.set_title("Jerarquía de Friedman\n🏆 = Mejor configuración")
    ax2.legend(fontsize=9)
    # Añadir etiquetas de valor
    for bar, val in zip(bars, mean_ranks):
        ax2.text(val + 0.05, bar.get_y() + bar.get_height()/2,
                 f"{val:.2f}", va="center", fontsize=8)
    ax2.tick_params(axis='y', labelsize=7)
    ax2.invert_yaxis()

    plt.tight_layout()
    plot_path = os.path.join(BASE_DIR, "statistical_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"  {plot_path}")
    print(f"\n{'='*60}\n")

# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not os.path.exists(TOP10_PATH):
        print(f"ERROR: No se encontró {TOP10_PATH}")
        print("Ejecuta primero: uv run python pruebas/ablation_study.py")
        sys.exit(1)

    data = load_top10(TOP10_PATH)
    print(f"Cargadas {len(data)} configuraciones del Top-10.")
    run_analysis(data)
