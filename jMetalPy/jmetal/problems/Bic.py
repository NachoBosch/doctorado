from jmetal.core.problem import BinaryProblem
from jmetal.core.solution import BinarySolution
import numpy as np
from typing import List
from dataclasses import dataclass
from scipy.stats import variation

@dataclass
class BiclusterResult:
    """Clase para almacenar los resultados de un bicluster"""
    rows: np.ndarray
    columns: np.ndarray
    msr: float  # Mean Squared Residue
    volume: int
    variance: float
    
class BiclusteringProblem:
    def __init__(self, data: np.ndarray, weights: dict = None, min_rows: int = 2, min_cols: int = 2):

        self.data = data if isinstance(data, np.ndarray) else data.to_numpy()
        self.number_of_variables = data.shape[0] + data.shape[1]
        self.number_of_objectives = 1
        self.number_of_constraints = 0
        
        # Valores por defecto para los pesos
        self.weights = weights or {
            'msr': 0.4,      # Mean Squared Residue
            'volume': 0.3,   # Tamaño del bicluster
            'variance': 0.2, # Varianza de los genes
            'overlap': 0.1   # Penalización por solapamiento
        }
        
        self.min_rows = min_rows
        self.min_cols = min_cols
        self.previous_biclusters: List[set] = []
        
    def evaluate(self, solution: BinarySolution) -> BinarySolution:
        """
        Evalúa una solución candidata.
        
        Args:
            solution: Solución binaria que representa un bicluster
        
        Returns:
            Solución evaluada con su valor de fitness
        """
        # Obtener índices de filas y columnas seleccionadas
        rows = np.arange(self.data.shape[0])
        cols = np.arange(self.data.shape[1])
        selected_rows = rows[solution.variables[:self.data.shape[0]]]
        selected_cols = cols[solution.variables[self.data.shape[0]:]]
        
        # Verificar tamaños mínimos
        if len(selected_rows) < self.min_rows or len(selected_cols) < self.min_cols:
            solution.objectives[0] = float('inf')
            return solution
            
        # Extraer submatriz
        submatrix = self.data[np.ix_(selected_rows, selected_cols)]
        
        # Calcular componentes del fitness
        fitness_components = self._calculate_fitness_components(submatrix, selected_rows)
        
        # Calcular fitness final
        fitness = (
            self.weights['msr'] * (1 / (fitness_components['msr'] + 1e-6)) +
            self.weights['volume'] * fitness_components['volume'] +
            self.weights['variance'] * fitness_components['variance'] -
            self.weights['overlap'] * fitness_components['overlap']
        )
        
        solution.objectives[0] = fitness  # Negativo porque jMetal minimiza por defecto
        self.previous_biclusters.append(set(selected_rows))
        return solution
    
    def _calculate_fitness_components(self, submatrix: np.ndarray, selected_rows: np.ndarray) -> dict:
        """
        Calcula los componentes individuales del fitness.
        
        Args:
            submatrix: Submatriz del bicluster
            selected_rows: Índices de las filas seleccionadas
            
        Returns:
            Diccionario con los componentes del fitness
        """
        # Mean Squared Residue (MSR)
        row_means = np.mean(submatrix, axis=1, keepdims=True)
        col_means = np.mean(submatrix, axis=0, keepdims=True)
        matrix_mean = np.mean(submatrix)
        residue = submatrix - row_means - col_means + matrix_mean
        msr = np.mean(residue ** 2)
        
        # Volumen normalizado (entre 0 y 1)
        max_volume = self.data.shape[0] * self.data.shape[1]
        volume = (submatrix.shape[0] * submatrix.shape[1]) / max_volume
        
        # Varianza de los genes
        gene_variance = np.mean(variation(submatrix, axis=1))
        
        # Penalización por solapamiento
        overlap = sum(len(set(selected_rows) & prev_cluster) 
                     for prev_cluster in self.previous_biclusters)
        max_overlap = len(selected_rows) * len(self.previous_biclusters)
        overlap_penalty = overlap / max_overlap if max_overlap > 0 else 0
        
        return {
            'msr': msr,
            'volume': volume,
            'variance': gene_variance,
            'overlap': overlap_penalty
        }
    
    def create_solution(self) -> BinarySolution:
        """
        Crea una solución inicial aleatoria.
        
        Returns:
            Nueva solución binaria
        """
        solution = BinarySolution(
            number_of_variables=self.number_of_variables,
            number_of_objectives=self.number_of_objectives
        )
        
        # Inicializar todos los genes a False
        solution.variables = [False] * self.number_of_variables
        
        # Seleccionar aleatoriamente un subconjunto de filas y columnas
        num_rows = self.data.shape[0]
        num_cols = self.data.shape[1]
        
        # Asegurar que se seleccionen al menos los mínimos requeridos
        n_rows = np.random.randint(self.min_rows, max(self.min_rows + 1, int(0.3 * num_rows)))
        n_cols = np.random.randint(self.min_cols, max(self.min_cols + 1, int(0.3 * num_cols)))
        
        selected_rows = np.random.choice(num_rows, size=n_rows, replace=False)
        selected_cols = np.random.choice(num_cols, size=n_cols, replace=False)
        
        # Activar los genes seleccionados
        for row in selected_rows:
            solution.variables[row] = True
        for col in selected_cols:
            solution.variables[num_rows + col] = True
        solution.objectives = [0]
        return solution
    
    def get_name(self) -> str:
        return "BiclusteringProblem"
    
    def get_bicluster_info(self, solution: BinarySolution) -> BiclusterResult:
        """
        Extrae información detallada del bicluster representado por una solución.
        
        Args:
            solution: Solución binaria que representa un bicluster
            
        Returns:
            Objeto BiclusterResult con la información del bicluster
        """
        rows = np.arange(self.data.shape[0])
        cols = np.arange(self.data.shape[1])
        selected_rows = rows[solution.variables[:self.data.shape[0]]]
        selected_cols = cols[solution.variables[self.data.shape[0]:]]
        
        submatrix = self.data[np.ix_(selected_rows, selected_cols)]
        fitness_components = self._calculate_fitness_components(submatrix, selected_rows)
        
        return BiclusterResult(
            rows=selected_rows,
            columns=selected_cols,
            msr=fitness_components['msr'],
            volume=submatrix.shape[0] * submatrix.shape[1],
            variance=fitness_components['variance']
        )