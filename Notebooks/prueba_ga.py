from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split,KFold
from sklearn.preprocessing import LabelEncoder
from jmetal.core.solution import BinarySolution
from jmetal.algorithm.singleobjective import GeneticAlgorithm
from jmetal.operator import BinaryTournamentSelection, SBXCrossover, BitFlipMutation, DifferentialEvolutionCrossover, PolynomialMutation, CXCrossover, SPXCrossover
from jmetal.util.termination_criterion import StoppingByEvaluations
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


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
    solution.objectives[0] = -fitness
    solution.constraints = []

  def create_solution(self):
      new_solution = BinarySolution(
          number_of_variables = self.number_of_variables,
          number_of_objectives = self.number_of_objectives,
          number_of_constraints = self.number_of_constraints
      )
      new_solution.variables = [True if np.random.rand() > 0.5 else False for _ in range(self.number_of_variables)]
      new_solution.objectives = [0 for _ in range(self.number_of_objectives)]
      new_solution.constraints = [0 for _ in range(self.number_of_constraints)]
      return new_solution

  def get_name(self):
    return "FeatureSelectionGA"
  
#DATA
df_hd = pd.read_csv('../Data/HD_filtered.csv')
encoder = LabelEncoder()
X = df_hd.drop(columns=['Samples','Grade']).to_numpy()
y = encoder.fit_transform(df_hd.Grade.to_numpy())
clases = list(df_hd.columns[:-2])

problem = FeatureSelectionGA(X,y,0.9)

algorithm = GeneticAlgorithm(
    problem=problem,
    population_size=100,
    offspring_population_size=100,
    mutation=BitFlipMutation(0.01),
    crossover=SPXCrossover(0.9),
    selection=BinaryTournamentSelection(),
    termination_criterion=StoppingByEvaluations(max_evaluations=10000)
)

algorithm.run()
test=3
soluciones_ls = algorithm.get_result()
objectives = soluciones_ls.objectives
variables = soluciones_ls.variables

var_squeezed = np.squeeze(variables)
genes_selected = [gen for gen,var in zip(clases,var_squeezed) if var]#==1]

with open(f'Resultados_FS_{test}.txt','w') as f:
    f.write(f"Name: {algorithm.get_name()}\n")
    f.write(f"Solucion objectives: {objectives}\n")
    f.write(f"Solucion variables: {variables}\n")
    f.write(f"Solucion variables type: {type(variables)}\n")
    f.write(f"Solucion variables amount: {len(variables)}\n")
    f.write(f"Selected genes: {genes_selected}\n")
    f.write(f"Selected genes amount: {len(genes_selected)}\n")