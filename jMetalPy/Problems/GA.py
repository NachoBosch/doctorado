from Solutions.solutions import BinarySolution
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split,KFold

import pandas as pd
import numpy as np



#PROBLEM
class FeatureSelectionGA():
  def __init__(self,X,y,alfa):
    self.X = X
    self.y = y
    self.alfa = alfa
    self.number_of_variables = X.shape[1]
    self.number_of_objectives = 1
    self.number_of_constraints = 0

  def evaluate(self, solution):
    selected_features = np.flatnonzero(solution.variables)
    X_selected = self.X[:, selected_features]

    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    scores = []
    model = SVC()
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
    solution.objectives[0] = 1-fitness
    solution.constraints = []

  def create_solution(self):
      new_solution = BinarySolution(
          number_of_variables = self.number_of_variables,
          number_of_objectives = self.number_of_objectives,
          number_of_constraints = self.number_of_constraints
      )
      # Genera un número aleatorio de características seleccionadas entre 10% y 90% del total
      # num_selected_features = np.random.randint(
      #     int(0.1 * self.number_of_variables), int(0.9 * self.number_of_variables) + 1
      # )
      # # Inicializa todas las variables a 0
      # new_solution.variables = [0] * self.number_of_variables
      # # Selecciona al azar las características que estarán activadas (1)
      # selected_indices = np.random.choice(range(self.number_of_variables), num_selected_features, replace=False)
      # for index in selected_indices:
      #   new_solution.variables[index] = 1
      new_solution.variables = [True if np.random.randint(0, 1) == 0 else False for _ in range(self.number_of_variables)]
      new_solution.objectives = [0 for _ in range(self.number_of_objectives)]
      new_solution.constraints = [0 for _ in range(self.number_of_constraints)]
      # print(new_solution.variables)
      return new_solution

  def get_name(self):
    return "FeatureSelectionGA"