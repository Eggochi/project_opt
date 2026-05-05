import sys
import os
import numpy as np

# Añadir el directorio raíz al path para importar src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.problemas import KnapsackProblem
from pymoo.operators.crossover.pntx import SinglePointCrossover
from src.CombinationMethods import PymooCrossoverCombination
from src.LocalSearchMethods import LocalSearchFirstImprovement
from src.RefSetMethods import ReferenceSet

def test_knp_reader():
    knp_file = "KNP/KNP-100-2objs.knp"
    if not os.path.exists(knp_file):
        # Intentar ruta relativa si se corre desde pruebas/
        knp_file = "../KNP/KNP-100-2objs.knp"
        if not os.path.exists(knp_file):
            print(f"Error: No se encontró el archivo {knp_file}")
            return

    # Probar con el primer objetivo (obj_idx=0)
    prob = KnapsackProblem(knp_file, obj_idx=0)
    
    print(f"Problema cargado: {knp_file}")
    print(f"Items: {prob.n_var}")
    print(f"Objetivo: {prob.obj_idx}")
    print(f"Capacidad: {prob.C:.2f}")
    
    # Evaluar una solución aleatoria (muchos items prendidos)
    X = np.random.randint(0, 2, (1, prob.n_var))
    # Asegurar que sea factible para ver un profit real sin penalización
    X_feasible = np.zeros((1, prob.n_var))
    X_feasible[0, :5] = 1 # Solo 5 items
    
    out = {}
    prob._evaluate(X, out)
    
    print(f"\nEvaluación de solución aleatoria (X shape {X.shape}):")
    print(f"F (con penalización): {out['F'][0]:.2f}")
    print(f"G (violación de restricción): {out['G'][0]:.2f}")
    
    out_f = {}
    prob._evaluate(X_feasible, out_f)
    print(f"\nEvaluación de solución factible (primeros 5 items):")
    print(f"F: {out_f['F'][0]:.2f}")
    print(f"G: {out_f['G'][0]:.2f}")
    
    # Comparar profits individuales
    total_p = np.sum(X_feasible[0] * prob.p)
    print(f"Profit esperado: {-total_p:.2f} (pymoo minimiza, profit es negativo)")

if __name__ == "__main__":
    test_knp_reader()
