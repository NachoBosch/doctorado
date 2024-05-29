from Solutions.solutions import BinarySolution
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from Algorithms import NeuralNet
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split,KFold

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
        kf = KFold(n_splits=4, shuffle=True, random_state=42)
        scores = []
        model = KNeighborsClassifier(6)
        for trainI, testI in kf.split(X_selected):
            X_train, X_test = X_selected[trainI], X_selected[testI]
            y_train, y_test = self.y[trainI], self.y[testI]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            scores.append(acc)

        acc_avg = np.mean(scores)

        num_variables = len(selected_features)
        beta = 1 - self.alfa
        fitness = 1.0 - (num_variables/self.X.shape[1]) # Primera parte de la función agregativa
        fitness = (self.alfa * fitness) + (beta * acc_avg)

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
        new_solution.variables = [np.random.randint(0, 2) for _ in range(self.number_of_variables)]
        new_solution.objectives = [0 for _ in range(self.number_of_objectives)]
        new_solution.constraints = [0 for _ in range(self.number_of_constraints)]
        # print(new_solution.variables)
        return new_solution

    def get_name(self):
        return "FeatureSelectionCGA"
