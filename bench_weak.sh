#!/bin/bash

# Configuración: el tamaño del problema crece con los procesos
BASE_N=150
EVALS=10000
REPEATS=30

# Lista de procesos a probar
PROCS=(1 2 4 8 16 32)

echo "Iniciando batería de pruebas de ESCALABILIDAD DÉBIL..."

for p in "${PROCS[@]}"
do
    echo "===================================================="
    echo "Ejecutando con $p procesos (N = $((BASE_N * p)))"
    echo "===================================================="
    mpirun -n $p python pruebas/escalabilidad.py --weak --base-n $BASE_N --evals $EVALS --repeats $REPEATS
done

echo "¡Pruebas de escalabilidad débil completadas!"
