#!/bin/bash

# Configuración: N se queda fijo para medir aceleración
N_FIXED=500
EVALS=10000
REPEATS=30

# Lista de procesos
PROCS=(1 2 4 8 16 32)

echo "Iniciando batería de pruebas de ESCALABILIDAD FUERTE..."

for p in "${PROCS[@]}"
do
    echo "===================================================="
    echo "Ejecutando con $p procesos | N fijo = $N_FIXED"
    echo "===================================================="
    mpirun -n $p python pruebas/escalabilidad.py --strong --n $N_FIXED --evals $EVALS --repeats $REPEATS
done

echo "¡Pruebas de escalabilidad fuerte completadas!"
