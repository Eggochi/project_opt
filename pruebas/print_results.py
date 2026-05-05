import pandas as pd
import os

def print_table():
    csv_file = "parallel_performance_results.csv"
    if not os.path.exists(csv_file):
        print(f"Error: No se encontró {csv_file}")
        return

    df = pd.read_csv(csv_file)
    df = df.sort_values('n_procs')

    # Obtener tiempo base (T1) para cálculos de speedup
    t1_row = df[df['n_procs'] == 1]
    t1 = t1_row['t_mean'].values[0] if not t1_row.empty else None

    print("\n" + "="*110)
    print(f"{'RESUMEN DE RENDIMIENTO PARALELO':^110}")
    print("="*110)
    
    header = f"{'Procs':>6} | {'Fitness (avg)':>15} | {'Tiempo (s)':>12} | {'Evals':>10} | {'Speedup':>10} | {'Eficiencia':>12}"
    print(header)
    print("-" * 110)

    for _, row in df.iterrows():
        p = int(row['n_procs'])
        f = row['f_mean']
        t = row['t_mean']
        e = row['e_mean']

        std_f = row['f_std']
        std_t = row['t_std']
        std_e = row['e_std']
        
        speedup = t1 / t if t1 else 1.0
        eff = (speedup / p) * 100
        
        line = f"{p:>6} | {f:>15.2f} (±{std_f:>6.2f}) | {t:>12.4f} (±{std_t:>6.4f}) | {e:>10.0f} (±{std_e:>6.0f}) | {speedup:>9.2f}x | {eff:>11.2f}%"
        print(line)

    print("="*110 + "\n")

if __name__ == "__main__":
    print_table()
