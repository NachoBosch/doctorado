from Solutions.solutions import BinarySolution
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np


class FeatureSelectionCGA():
    def __init__(self,X,y,alfa):
        self.X = X
        self.y = y
        self.alfa = alfa
        self.number_of_variables = X.shape[1]
        self.number_of_objectives = 1
        self.number_of_constraints = 0
        self.GenesSelected = []
        self.FitnessValues = []

    def evaluate(self, solution):
        selected_features = np.flatnonzero(solution.variables)
        if len(selected_features) == 0:
            solution.objectives[0] = 0
            return solution
        X_selected = self.X[:, selected_features]

        Xtrain,Xtest,ytrain,ytest = train_test_split(X_selected,self.y, stratify=self.y)

        model = RandomForestClassifier()
        model.fit(Xtrain, ytrain)
        y_pred = model.predict(Xtest)
        acc = accuracy_score(ytest, y_pred)

        num_variables = len(selected_features)
        beta = 1 - self.alfa
        fitness = 1.0 - (num_variables/self.X.shape[1]) # Primera parte de la función agregativa
        fitness = (self.alfa * fitness) + (beta * acc)
        # print(solution.objectives)
        solution.objectives[0] = -fitness
        solution.constraints = []
        self.GenesSelected.append(num_variables)
        self.FitnessValues.append(fitness)
        return solution

    def create_solution(self):
        new_solution = BinarySolution(
            number_of_variables = self.number_of_variables,
            number_of_objectives = self.number_of_objectives,
            number_of_constraints = self.number_of_constraints
        )
        #[np.random.randint(0, 2, size=1).tolist() for _ in range(self.number_of_variables)]
        new_solution.variables = [np.random.randint(0, 2) for _ in range(self.number_of_variables)]
        new_solution.objectives = [0 for _ in range(self.number_of_objectives)]
        new_solution.constraints = [0 for _ in range(self.number_of_constraints)]
        print(new_solution.variables)
    
        return new_solution

    def get_name(self):
        return "FeatureSelectionCGA"
  
    #Solution Function
    # def get_genes(self,selected_variables):
    #     genes_selected = [gen for gen, var in zip(clases, selected_variables) if var == 1]
    #     return len(genes_selected)
    # def evaluate_and_save(self,algorithm):
    #     fitness_data = []
    #     genes_data = []
    #     solutions = algorithm.get_result()
    #     fitness_data.append(solutions.objectives)
    #     genes_data.append(get_genes(np.squeeze(solutions.variables)))
    #     pd.DataFrame({'Objective': fitness_data, 
    #                     'Genes': genes_data}).to_excel('Resultados_CGA/Resultados_func_agregativa/Experimento1/resultados.xlsx', 
    #                                                 index=False)
    # print(GenesSelected)
    # print(FitnessValues)
    # pd.DataFrame({'Objective': FitnessValues, 
    #             'Genes': GenesSelected}).to_excel('Resultados_CGA/Resultados_func_agregativa/Experimento1/resultados_2.xlsx')