import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.problemas import QUBO
from src.config import config
import matplotlib.pyplot as plt
import numpy as np
import random
from itertools import product
import os

'''''
Distribucion de espacio de soluciones de QUBO aleatorio
'''''
DIR = config.get("experiment", "output_dir", default="Graficas") + "/distribucion"
os.makedirs(DIR, exist_ok=True)

np.random.seed(config.get("problem", "seed", default=42))

n_list = [10, 15, 20] # Reduced for testing as product grows exponentially

for i in n_list:
    #Q aleatoria centrada en 0
    Q = np.random.randint(-8, 8, size=(i, i))
    Q = (Q + Q.T) // 2

    #Q aleatoria con sesgo a negativos
    Q2 = np.random.randint(-3, 8, size=(i, i))
    Q2 = (Q2 + Q2.T) // 2

    #Q con posible minimo forzado 
    Q3=Q2.copy()
    for j in range(0, i, 2):
        for k in range(0, i, 2):
            Q3[j, k] = -4
            Q3[k, j] = -4
        Q3[j, j] = -6

    problem_qubo = QUBO(Q)
    problem_qubo2 = QUBO(Q2)
    problem_qubo3 = QUBO(Q3)

    x_vals = np.array(list(product([0, 1], repeat=i)))

    
    # Función aleatoria
    out1 = {}
    problem_qubo._evaluate(x_vals, out1)
    y_vals_aleatorio = out1['F']
    
    # Función estructurada (regular)
    out2 = {}
    problem_qubo2._evaluate(x_vals, out2)
    y_vals_aleatorio2 = out2['F']

    # Función estructurada (regular)
    out3 = {}
    problem_qubo3._evaluate(x_vals, out3)
    y_vals_forzado = out3['F']

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))

    ax1.hist(y_vals_aleatorio, bins=20)
    ax1.set_title(f'Distribución de espacio de soluciones para QUBO Aleatorio (n={i})')
    ax1.set_xlabel('Valor de la Función (F)')
    ax1.set_ylabel('Frecuencia')
    ax1.grid(True)
    
    ax2.hist(y_vals_aleatorio2, bins=20)
    ax2.set_title(f'Distribución de espacio de soluciones para QUBO Aleatorio 2 (n={i})')
    ax2.set_xlabel('Valor de la Función (F)')
    ax2.set_ylabel('Frecuencia')
    ax2.grid(True)

    ax3.hist(y_vals_forzado, bins=20)
    ax3.set_title(f'Distribución de espacio de soluciones para QUBO Forzado (n={i})')
    ax3.set_xlabel('Valor de la Función (F)')
    ax3.set_ylabel('Frecuencia')
    ax3.grid(True)

    plt.savefig(f'{DIR}/qubo_distribucion_n_{i}.png')
    plt.close()

    print(f"\nResultados para n={i}:")
    print(f"Minimo: {np.min(y_vals_aleatorio)}")
    print(f"Maximo: {np.max(y_vals_aleatorio)}")
    print(f"Media: {np.mean(y_vals_aleatorio)}")
    print(f"Desviacion estandar: {np.std(y_vals_aleatorio)}")
    print("-"*20)
    print(f"Minimo: {np.min(y_vals_aleatorio2)}")
    print(f"Maximo: {np.max(y_vals_aleatorio2)}")
    print(f"Media: {np.mean(y_vals_aleatorio2)}")
    print(f"Desviacion estandar: {np.std(y_vals_aleatorio2)}")
    print(f"X optimas: {x_vals[np.argmin(y_vals_aleatorio2)]}")
    print("-"*20)
    print(f"Minimo: {np.min(y_vals_forzado)}")
    print(f"Maximo: {np.max(y_vals_forzado)}")
    print(f"Media: {np.mean(y_vals_forzado)}")
    print(f"Desviacion estandar: {np.std(y_vals_forzado)}")
    print(f"X optimas: {x_vals[np.argmin(y_vals_forzado)]}")
    print("-"*20)

