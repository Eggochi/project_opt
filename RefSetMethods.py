import numpy as np
from scipy.spatial.distance import cdist

import numpy as np
from scipy.spatial.distance import cdist

class ReferenceSet:
    def __init__(self, diversity_measure):
        self.diversity_measure = diversity_measure
        self.RefSet = []
        # seen_solutions debe servir para no procesar duplicados en el pool, 
        # no para expulsar soluciones del RefSet.
        self.seen_solutions = set()

    def create(self, solution_pool, reference_set_size, diversity_threshold):
        """Crea el conjunto de referencia permitiendo la persistencia de los mejores."""
        # 1. Filtramos duplicados EXACTOS dentro del pool actual para no trabajar de más
        # Pero permitimos que lo que ya estaba en el RefSet sea elegible
        unique_pool = []
        temp_seen = set()
        for s in solution_pool:
            identificador = tuple(s.X)
            if identificador not in temp_seen:
                unique_pool.append(s)
                temp_seen.add(identificador)
        
        X_pool, F_pool = self._unpack_pool(unique_pool)
        L = X_pool.shape[1]
        
        quality_size = reference_set_size // 2
        available_mask = np.ones(len(unique_pool), dtype=bool)
        ref_indices = []

        # 2. SELECCIÓN POR CALIDAD
        # Aquí NO filtramos por self.seen_solutions, porque queremos que la mejor 
        # solución sobreviva generación tras generación.
        quality_indices = np.argsort(F_pool)
        for idx in quality_indices:
            if len(ref_indices) >= quality_size:
                break
            
            # Solo verificamos diversidad respecto a los ya elegidos en ESTA ronda
            if self._is_diverse_enough(X_pool[idx], X_pool[ref_indices], L, diversity_threshold):
                ref_indices.append(idx)
                available_mask[idx] = False

        # 3. SELECCIÓN POR DIVERSIDAD
        ref_indices = self._fill_by_diversity(
            X_pool, ref_indices, available_mask, 
            reference_set_size, diversity_threshold, L
        )

        self.RefSet = [unique_pool[i] for i in ref_indices]
        
        # Opcional: Registrar en la memoria histórica lo que entró al RefSet
        for s in self.RefSet:
            self.seen_solutions.add(tuple(s.X))

        return self.RefSet, None

    def update(self, new_solutions, reference_set_size, diversity_threshold):
        """Actualiza el RefSet permitiendo que las mejores soluciones antiguas compitan."""
        old_ref_set = list(self.RefSet)
        
        # Filtramos las new_solutions: si una solución nueva es idéntica a algo
        # que ya procesamos históricamente Y no está en el RefSet actual, la ignoramos.
        filtered_new = [
            s for s in new_solutions 
            if tuple(s.X) not in self.seen_solutions or s in old_ref_set
        ]
        
        combined_pool = old_ref_set + filtered_new
        
        # Re-creamos el RefSet. Las mejores soluciones de 'old_ref_set' 
        # ahora pueden ganar su lugar de nuevo.
        new_ref_set, _ = self.create(combined_pool, reference_set_size, diversity_threshold)
        
        added_solutions = [s for s in new_ref_set if s not in old_ref_set]
        return added_solutions, old_ref_set
    # --- Métodos de apoyo (Privados) ---

    def _unpack_pool(self, pool):
        """Extrae vectores X y valores F de la lista de soluciones."""
        X = np.array([s.X for s in pool])
        F = np.array([float(s.F[0]) if isinstance(s.F, np.ndarray) else float(s.F) for s in pool])
        return X, F

    def _is_diverse_enough(self, candidate_x, current_ref_x, L, threshold):
        """Verifica si un candidato es suficientemente distinto a lo ya seleccionado."""
        if len(current_ref_x) == 0:
            return True
        
        dists = cdist(candidate_x.reshape(1, -1), current_ref_x, metric=self.diversity_measure) * L
        return np.all(dists >= threshold)

    def _fill_by_diversity(self, X_pool, ref_indices, available_mask, target_size, threshold, L):
        """Completa el RefSet buscando los candidatos más alejados posibles (Max-Min)."""
        if len(ref_indices) == 0:
            return ref_indices

        # Calcular distancias iniciales de todos hacia el RefSet actual
        dists_matrix = cdist(X_pool, X_pool[ref_indices], metric=self.diversity_measure) * L
        min_dists = np.min(dists_matrix, axis=1)

        while len(ref_indices) < target_size:
            if not np.any(available_mask):
                break
            
            # Elegir el que tiene la mayor "distancia mínima" (el más lejano)
            best_idx = np.where(available_mask, min_dists, -1).argmax()

            if best_idx == -1:
                break
            
            if tuple(X_pool[best_idx]) not in self.seen_solutions:
                ref_indices.append(best_idx)
                self.seen_solutions.add(tuple(X_pool[best_idx]))
                
                # Actualizar distancias mínimas considerando al nuevo integrante
                new_dists = cdist(X_pool, X_pool[best_idx].reshape(1, -1), metric=self.diversity_measure).flatten() * L
                min_dists = np.minimum(min_dists, new_dists)
            
            # SIEMPRE marcamos como no disponible
            available_mask[best_idx] = False
            
        return ref_indices