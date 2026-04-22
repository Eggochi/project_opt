import numpy as np
import matplotlib.pyplot as plt
import itertools
from problemas import NKLandscape
import os

os.makedirs('nk_landscapes', exist_ok=True)

N_values = [4, 8, 12] # Bajé 16 a 12 para que sea más rápido de visualizar, puedes volver a subirlo
percentages = [0.25, 0.5, 0.75, 1.0]

def gray_decode(n):
    """Convierte un código binario a su equivalente en Gray para mantener vecindad."""
    mask = n >> 1
    while mask != 0:
        n = n ^ mask
        mask = mask >> 1
    return n

def plot_landscape(N, K, percentage):
    print(f"Generating landscape for N={N}, K={K} ({int(percentage*100)}%)")
    problem = NKLandscape(N, K)
    
    # 1. Generación exhaustiva (Solo viable para N < 20)
    all_combinations = list(itertools.product([0, 1], repeat=N))
    X = np.array(all_combinations)
    
    # 2. Evaluación vectorizada
    out = {}
    problem._evaluate(X, out)
    
    # Pymoo suele minimizar (F es positivo), lo negamos para visualizar "montañas" de éxito
    Z_flat = -out['F'].flatten()
    
    # 3. Mapeo a rejilla 2D usando Gray Code
    half_N = N // 2
    rem_N = N - half_N
    
    Z_grid = np.zeros((2**rem_N, 2**half_N))
    
    for i, combination in enumerate(all_combinations):
        x_bits = combination[:half_N]
        y_bits = combination[half_N:]
        
        # Corrección: Conversión de bits más eficiente
        x_val_bin = int("".join(map(str, map(int, x_bits))), 2)
        y_val_bin = int("".join(map(str, map(int, y_bits))), 2)
        
        # Aplicamos Gray para que la cercanía en la rejilla sea cercanía en Hamming
        x_val = gray_decode(x_val_bin)
        y_val = gray_decode(y_val_bin)
        
        Z_grid[y_val, x_val] = Z_flat[i]
        
    # 4. Configuración de la visualización
    x_range = np.arange(2**half_N)
    y_range = np.arange(2**rem_N)
    x_mesh, y_mesh = np.meshgrid(x_range, y_range)
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Corrección visual: rstride y cstride evitan que la gráfica se vea saturada en N grandes
    stride = 1 if N <= 8 else 2
    surf = ax.plot_surface(x_mesh, y_mesh, Z_grid, cmap='terrain', 
                           edgecolor='none', rstride=stride, cstride=stride,
                           antialiased=True)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Fitness (Negated)')
    
    ax.set_title(f"NK Landscape Surface\nN={N}, K={K} ({int(percentage*100)}% Ruggedness)", 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Gray Code X (First half bits)')
    ax.set_ylabel('Gray Code Y (Second half bits)')
    ax.set_zlabel('Fitness')
    
    # Ajustar ángulo de cámara para mejor perspectiva
    ax.view_init(elev=30, azim=45)
    
    filename = f"nk_landscapes/NK_N{N}_K{K}_pct{int(percentage*100)}.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")

# Ejecución principal
for N in N_values:
    for percentage in percentages:
        # K es el número de interacciones. K=0 es suave, K=N-1 es máximo ruido.
        K = int(round((N - 1) * percentage))
        K = max(0, min(N - 1, K))
        plot_landscape(N, K, percentage)