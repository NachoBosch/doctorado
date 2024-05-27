from Solutions.solutions import BinarySolution
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

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

    Xtrain,Xtest,ytrain,ytest = train_test_split(X_selected,self.y,test_size=0.3,random_state=42, stratify=self.y)

    model = SVC()
    model.fit(Xtrain, ytrain)
    y_pred = model.predict(Xtest)
    acc = accuracy_score(ytest, y_pred)

    num_variables = len(selected_features)
    beta = 1 - self.alfa
    fitness = 1.0 - (num_variables/self.X.shape[1]) # Primera parte de la función agregativa
    fitness = (self.alfa * fitness) + (beta * acc)
    solution.objectives[0] = -fitness
    solution.constraints = []

  def create_solution(self):
    new_solution = BinarySolution(
        number_of_variables = self.number_of_variables,
        number_of_objectives = self.number_of_objectives,
        number_of_constraints = self.number_of_constraints
    )
    
    new_solution.variables = [np.random.randint(0, 2) for _ in range(self.number_of_variables)]
    new_solution.objectives = [0 for _ in range(self.number_of_objectives)]
    new_solution.constraints = [0 for _ in range(self.number_of_constraints)]

    return new_solution

  def get_name(self):
    return "FeatureSelectionGA"