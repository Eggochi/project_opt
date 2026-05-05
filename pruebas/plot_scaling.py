import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_strong_scaling(csv_path="strong_scaling_stats.csv", output_path="strong_scaling_plot.png"):
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Speedup plot (Left Y-axis)
    color = 'tab:blue'
    ax1.set_xlabel('Number of Processors (n_procs)', fontsize=12)
    ax1.set_ylabel('Speedup', color=color, fontsize=12)
    ax1.plot(df['n_procs'], df['speedup'], marker='o', linewidth=2, color=color, label='Actual Speedup')
    
    # Ideal Speedup reference line
    ax1.plot(df['n_procs'], df['n_procs'], linestyle='--', color='gray', label='Ideal Speedup')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(df['n_procs'])
    ax1.set_ylim(bottom=0)

    # Efficiency plot (Right Y-axis)
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Efficiency (%)', color=color, fontsize=12)
    ax2.plot(df['n_procs'], df['efficiency'], marker='s', linestyle='-', linewidth=2, color=color, label='Efficiency')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 110)

    # Add legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    plt.title(f'Strong Scaling Performance (Problem Size N={df["n_problem"].iloc[0]})', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    fig.tight_layout()
    
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved successfully to {output_path}")

def plot_weak_scaling(csv_path="weak_scaling_stats.csv", output_path="weak_scaling_plot.png"):
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    # Use a premium style
    plt.style.use('seaborn-v0_8-muted')
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Time plot (Left Y-axis) - Ideally should be flat
    color_time = '#2c3e50'  # Deep blue-gray
    ax1.set_xlabel('Number of Processors (n_procs)', fontsize=12, fontweight='600')
    ax1.set_ylabel('Execution Time (s)', color=color_time, fontsize=12, fontweight='600')
    
    # Plot mean time with error bars for min/max
    ax1.errorbar(df['n_procs'], df['mean'], yerr=[df['mean'] - df['min'], df['max'] - df['mean']], 
                 marker='o', markersize=8, linewidth=2, color=color_time, label='Mean Execution Time',
                 capsize=5, capthick=1.5, elinewidth=1.5)
    
    ax1.tick_params(axis='y', labelcolor=color_time)
    ax1.set_xticks(df['n_procs'])
    ax1.set_ylim(0, df['max'].max() * 1.2)
    
    # Annotate problem sizes
    for i, txt in enumerate(df['n_problem']):
        ax1.annotate(f"N={txt}", (df['n_procs'].iloc[i], df['mean'].iloc[i]), 
                     xytext=(0, 10), textcoords='offset points', ha='center', fontsize=9, fontweight='bold')

    # Efficiency plot (Right Y-axis)
    ax2 = ax1.twinx()
    color_eff = '#e67e22'  # Vibrant orange
    ax2.set_ylabel('Weak Efficiency', color=color_eff, fontsize=12, fontweight='600')
    ax2.plot(df['n_procs'], df['weak_efficiency'], marker='D', markersize=8, linestyle='-', 
             linewidth=3, color=color_eff, label='Weak Efficiency')
    
    # Ideal efficiency line
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Ideal Efficiency (1.0)')
    
    ax2.tick_params(axis='y', labelcolor=color_eff)
    ax2.set_ylim(0, max(1.1, df['weak_efficiency'].max() * 1.1))

    # Title and styling
    plt.title('Weak Scaling Performance: Constant Workload per Processor', fontsize=16, fontweight='bold', pad=20)
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', frameon=True, shadow=True)

    fig.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Weak scaling plot saved successfully to {output_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Strong Scaling
    strong_csv = os.path.join(base_dir, "strong_scaling_stats.csv")
    strong_out = os.path.join(base_dir, "strong_scaling_plot.png")
    plot_strong_scaling(strong_csv, strong_out)
    
    # Weak Scaling
    weak_csv = os.path.join(base_dir, "weak_scaling_stats.csv")
    weak_out = os.path.join(base_dir, "weak_scaling_plot.png")
    plot_weak_scaling(weak_csv, weak_out)
