#!/bin/bash

# --- Configuración Visual ---
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
BOLD='\033[1m'

echo -e "${BLUE}${BOLD}==================================================${NC}"
echo -e "${BLUE}${BOLD}      VALIDACIÓN DE ESCALABILIDAD Y PARALELISMO   ${NC}"
echo -e "${BLUE}${BOLD}==================================================${NC}"

# Parámetros por defecto
N_PROBLEM=${1:-50}
N_GENS=${2:-25}
N_TRIALS=${3:-10}
PROCS=(1 2 4 8 16)

echo -e "${BLUE}Configuración:${NC}"
echo -e "  - Tamaño del problema (N): ${BOLD}$N_PROBLEM${NC}"
echo -e "  - Número de generaciones (Gens): ${BOLD}$N_GENS${NC}"
echo -e "  - Número de pruebas: ${BOLD}$N_TRIALS${NC}"
echo -e "  - Procesos a probar: ${BOLD}${PROCS[*]}${NC}"
echo ""

# Verificar si mpirun existe
if ! command -v mpirun &> /dev/null
then
    echo -e "${BOLD}ERROR: mpirun no está instalado o no está en el PATH.${NC}"
    exit 1
fi

# Ejecutar validaciones
for P in "${PROCS[@]}"
do
    echo -e "${BLUE}>>> Probando con ${BOLD}$P${NC}${BLUE} procesos...${NC}"
    mpirun --oversubscribe -np $P uv run python pruebas/validar_paralelismo.py $N_PROBLEM $N_GENS $N_TRIALS

    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Prueba con $P procesos finalizada con éxito.${NC}"
    else
        echo -e "${BOLD}ERROR en la prueba con $P procesos.${NC}"
    fi
    echo -e "${BLUE}--------------------------------------------------${NC}"
done

echo -e "${GREEN}${BOLD}Validación completada.${NC}"
