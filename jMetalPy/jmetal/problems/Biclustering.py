from jmetal.core.solution import BinarySolution
import numpy as np
from scipy.stats import variation

class BiclusteringProblem:
    def __init__(self, data, alfa):
        self.data = data
        self.alfa = alfa
        self.number_of_variables = data.shape[0] + data.shape[1]
        self.number_of_objectives = 1
        self.number_of_constraints = 0
        self.previous_biclusters = []

    def evaluate(self, solution):
        data_array = self.data.to_numpy()

        selected_rows = np.flatnonzero(solution.variables[:data_array.shape[0]])
        selected_cols = np.flatnonzero(solution.variables[data_array.shape[0]:])
        
        if len(selected_rows) == 0 or len(selected_cols) == 0:
            solution.objectives[0] = float('inf')
            return solution
        
        submatrix = data_array[np.ix_(selected_rows, selected_cols)]
        
        row_means = np.mean(submatrix, axis=1, keepdims=True)
        col_means = np.mean(submatrix, axis=0, keepdims=True)
        matrix_mean = np.mean(submatrix)
        residue = submatrix - row_means - col_means + matrix_mean
        mean_squared_residue = np.mean(residue ** 2)

        volume = len(selected_rows) * len(selected_cols)

        gene_variance = np.mean(variation(submatrix, axis=1))
        overlap_penalty = self.calculate_overlap_penalty(selected_rows)

        # Ajustar el fitness
        w1, w2, w3, w4 = 0.4, 0.3, 0.2, 0.3  # Pesos
        fitness = (w1 * (1 / (mean_squared_residue + 1e-6)) +
                   w2 * volume +
                   w3 * gene_variance -
                   w4 * overlap_penalty)  # Minimizar genes en común

        solution.objectives[0] = fitness
        self.previous_biclusters.append(set(selected_rows))  # Registrar genes de bicluster
        return solution

        # w1, w2, w3 = 0.4, 0.4, 0.2
        # fitness = w1 * (1 / (mean_squared_residue + 1e-6)) + w2 * volume + w3 * gene_variance
        # # print(fitness)
        # solution.objectives[0] = fitness
        # return solution
    def calculate_overlap_penalty(self, selected_rows):
        # Calcular genes en común entre el bicluster actual y biclusters previos
        overlap_count = sum(len(set(selected_rows) & previous) for previous in self.previous_biclusters)
        return overlap_count


    def create_solution(self):
        new_solution = BinarySolution(
            number_of_variables=self.number_of_variables,
            number_of_objectives=self.number_of_objectives
        )

        new_solution.variables = [False] * self.number_of_variables
        num_rows = self.data.shape[0]
        num_cols = self.data.shape[1]
        selected_rows = np.random.choice(num_rows, size=int(0.1 * num_rows), replace=False)
        selected_cols = np.random.choice(num_cols, size=int(0.1 * num_cols), replace=False)

        for row in selected_rows:
            new_solution.variables[row] = True

        for col in selected_cols:
            new_solution.variables[num_rows + col] = True

        # Inicializar los objetivos
        new_solution.objectives = [0]
        return new_solution

    def get_name(self):
        return "BiclusteringProblem"
