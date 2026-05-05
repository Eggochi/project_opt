import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def interpolate_convergence(df, n_points=200):
    curves = []

    # Crear un ID de trial único, útil si el CSV tiene múltiples ejecuciones pegadas (append)
    if 'unique_trial' not in df.columns:
        df['unique_trial'] = (df['iteration'] == 0).cumsum()

    for n_proc in sorted(df['n_procs'].unique()):

        proc_df = df[df['n_procs'] == n_proc]

        # Máximo eval común entre todos los trials únicos de este n_proc
        max_common_eval = proc_df.groupby('unique_trial')['evals'].max().min()

        # Grid uniforme para interpolación
        eval_grid = np.linspace(
            proc_df['evals'].min(),
            max_common_eval,
            n_points
        )

        trial_curves = []

        for trial in sorted(proc_df['unique_trial'].unique()):

            trial_df = (
                proc_df[proc_df['unique_trial'] == trial]
                .sort_values('evals')
                .drop_duplicates('evals', keep='last')
            )

            interp_fit = np.interp(
                eval_grid,
                trial_df['evals'],
                trial_df['fitness']
            )

            trial_curves.append(interp_fit)

        trial_curves = np.array(trial_curves)

        curves.append(pd.DataFrame({
            'n_procs': n_proc,
            'evals': eval_grid,
            'mean': trial_curves.mean(axis=0),
            'std': trial_curves.std(axis=0)
        }))

    return pd.concat(curves, ignore_index=True)


def plot_convergence():
    csv_file = "parallel_convergence_history.csv"

    if not os.path.exists(csv_file):
        print(f"Error: No se encontró {csv_file}")
        return

    column_names = ['n_procs', 'trial', 'iteration', 'evals', 'fitness']

    try:
        df = pd.read_csv(csv_file)
        first_val = str(df.columns[0])
        if first_val.isdigit() or first_val == '1':
            df = pd.read_csv(csv_file, names=column_names)
    except Exception:
        df = pd.read_csv(csv_file, names=column_names)

    plt.style.use('seaborn-v0_8-muted')
    sns.set_context("talk")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    palette = sns.color_palette("viridis", n_colors=df['n_procs'].nunique())

    # ==========================================================
    # INTERPOLACIÓN CORRECTA DE CURVAS
    # ==========================================================
    df_avg = interpolate_convergence(df)

    limit=10000

    for i, nproc in enumerate(sorted(df_avg['n_procs'].unique())):
        subset = df_avg[df_avg['n_procs'] == nproc]

        ax1.plot(
            subset['evals'][subset['evals'] <=limit],
            subset['mean'][subset['evals'] <=limit],
            label=f'{nproc} proc(s)',
            color=palette[i],
            linewidth=2.5
        )

        ax1.fill_between(
            subset['evals'][subset['evals'] <=limit],
            subset['mean'][subset['evals'] <=limit] - subset['std'][subset['evals'] <=limit],
            subset['mean'][subset['evals'] <=limit] + subset['std'][subset['evals'] <=limit],
            color=palette[i],
            alpha=0.2
        )

    ax1.set_title('Curvas de Convergencia Promedio', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Evaluaciones Totales (MPI Sum)', fontsize=13)
    ax1.set_ylabel('Mejor Fitness Encontrado', fontsize=13)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(title='Procesadores')

    # ==========================================================
    # FITNESS FINAL ROBUSTO
    # ==========================================================
    final_fits = (
        df.sort_values('evals')
        .groupby(['n_procs', 'unique_trial'])
        .tail(1)
    )

    sns.boxplot(
        data=final_fits,
        x='n_procs',
        y='fitness',
        hue='n_procs',
        ax=ax2,
        palette=palette,
        legend=False
    )

    for patch in ax2.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.3))

    sns.stripplot(
        data=final_fits,
        x='n_procs',
        y='fitness',
        hue='n_procs',
        ax=ax2,
        palette=palette,
        size=8,
        jitter=True,
        edgecolor='gray',
        linewidth=1,
        legend=False
    )

    ax2.set_title('Distribución de Solución Final', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Número de Procesadores', fontsize=13)
    ax2.set_ylabel('Fitness Final', fontsize=13)
    ax2.grid(True, axis='y', linestyle=':', alpha=0.7)

    plt.suptitle(
        "Convergencia Paralela y Calidad de Soluciones",
        fontsize=22,
        fontweight='bold',
        y=1.03
    )

    plt.tight_layout()
    output_path = "parallel_convergence_plot.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Gráfico guardado en {output_path}")


if __name__ == "__main__":
    plot_convergence()