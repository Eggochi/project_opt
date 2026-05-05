import numpy as np
from scipy.spatial.distance import cdist

class ReferenceSet:
    def __init__(self, diversity_measure):
        self.diversity_measure = diversity_measure
        self.RefSet = []
        self.distancia_entre_soluciones=0
        

    def create(self, solution_pool, reference_set_size, diversity_threshold=None, quality_size=0.5):
        """Crea el conjunto de referencia permitiendo la persistencia de los mejores."""
        unique_pool = []
        temp_seen = set()
        for s in solution_pool:
            key = tuple(s.X)
            if key not in temp_seen:
                unique_pool.append(s)
                temp_seen.add(key)
        
        X_pool, F_pool = self._unpack_pool(unique_pool)
        L = X_pool.shape[1]
        
        quality_size = int(reference_set_size * quality_size)
        available_mask = np.ones(len(unique_pool), dtype=bool)
        ref_indices = []


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
        

        return self.RefSet, None

    def update(self, new_solutions, reference_set_size, diversity_threshold=None):
        """Actualiza el RefSet permitiendo que las mejores soluciones antiguas compitan."""
        old_ref_set = list(self.RefSet)
        combined_pool = old_ref_set + new_solutions
        new_ref_set, _ = self.create(combined_pool, reference_set_size, diversity_threshold, quality_size=0.5)
        
        added_solutions = [s for s in new_ref_set if s not in old_ref_set]
        old_solutions = [s for s in new_ref_set if s in old_ref_set]
        return added_solutions, old_solutions
    # --- Métodos de apoyo (Privados) ---

    def _unpack_pool(self, pool):
        """Extrae vectores X y valores F de la lista de soluciones."""
        X = np.array([s.X for s in pool])
        F = np.array([float(s.F[0]) if isinstance(s.F, np.ndarray) else float(s.F) for s in pool])
        return X, F

    def _is_diverse_enough(self, candidate_x, current_ref_x, L, threshold=None):
        """Verifica si un candidato es suficientemente distinto a lo ya seleccionado."""
        if len(current_ref_x) == 0 or threshold is None:
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
        self.distancia_entre_soluciones= min_dists

        while len(ref_indices) < target_size:
            if not np.any(available_mask):
                break
            
            # Elegir el que tiene la mayor "distancia mínima" (el más lejano)
            best_idx = np.where(available_mask, min_dists, -1).argmax()

            if best_idx == -1:
                break
            
            ref_indices.append(best_idx)
            
            # Actualizar distancias mínimas considerando al nuevo integrante
            new_dists = cdist(X_pool, X_pool[best_idx].reshape(1, -1), metric=self.diversity_measure).flatten() * L
            min_dists = np.minimum(min_dists, new_dists)
            
            # SIEMPRE marcamos como no disponible
            available_mask[best_idx] = False
            
        return ref_indices