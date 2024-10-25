from jmetal.core.solution import BinarySolution
import numpy as np
from scipy.stats import variation

class BiclusteringProblem:
    def __init__(self, data, alfa):
        self.data = data
        self.alfa = alfa
        self.number_of_variables = data.shape[0] + data.shape[1]  # Filas + Columnas
        self.number_of_objectives = 1
        self.number_of_constraints = 0

    def evaluate(self, solution):
        # Separar las filas y columnas seleccionadas basadas en las variables binarias
        selected_rows = np.flatnonzero(solution.variables[:self.data.shape[0]])
        selected_cols = np.flatnonzero(solution.variables[self.data.shape[0]:])
        
        if len(selected_rows) == 0 or len(selected_cols) == 0:
            solution.objectives[0] = float('inf')  # Penalización si no se seleccionan filas o columnas
            return solution
        
        # Submatriz del bicluster seleccionado
        submatrix = self.data[np.ix_(selected_rows, selected_cols)]
        
        # Calcular el residuo cuadrático medio
        row_means = np.mean(submatrix, axis=1, keepdims=True)
        col_means = np.mean(submatrix, axis=0, keepdims=True)
        matrix_mean = np.mean(submatrix)
        residue = submatrix - row_means - col_means + matrix_mean
        mean_squared_residue = np.mean(residue ** 2)

        # Calcular el volumen
        volume = len(selected_rows) * len(selected_cols)

        # Calcular la varianza génica
        gene_variance = np.mean(variation(submatrix, axis=1))

        # Fitness: combinamos los componentes del fitness (ajusta los pesos según corresponda)
        w1, w2, w3 = 0.4, 0.4, 0.2
        fitness = w1 * (1 / (mean_squared_residue + 1e-6)) + w2 * volume + w3 * gene_variance

        # Ajuste con el factor alfa
        # fitness = self.alfa * fitness
        solution.objectives[0] = -fitness
        return solution

    def create_solution(self):
        new_solution = BinarySolution(
            number_of_variables=self.number_of_variables,
            number_of_objectives=self.number_of_objectives)
        
        # Inicializamos las variables binarias (True/False)
        new_solution.variables = [True if np.random.rand() > 0.5 else False for _ in range(self.number_of_variables)]
        new_solution.objectives = [0]

        return new_solution

    def get_name(self):
        return "BiclusteringProblem"
