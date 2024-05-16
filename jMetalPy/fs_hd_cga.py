import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np


import sys
sys.path.insert(1, 'D:/Doctorado/doctorado/jMetalPy/jmetal')

from jmetal.core.solution import BinarySolution
from jmetal.algorithm.singleobjective import CellularGeneticAlgorithm
from jmetal.algorithm.multiobjective import NSGAII
from jmetal.operator import BinaryTournamentSelection, SBXCrossover, BitFlipMutation, DifferentialEvolutionCrossover, PolynomialMutation, CXCrossover, SPXCrossover
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.operator.selection import BestSolutionSelection


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif

#PROBLEM
class FeatureSelectionProblem():
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

    Xtrain,Xtest,ytrain,ytest = train_test_split(X_selected,self.y, stratify=self.y)

    model = RandomForestClassifier()
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
    
    new_variables = [np.random.randint(0, 2, size=1).tolist() for _ in range(self.number_of_variables)]
    # print(new_variables)
    new_solution.variables = new_variables
    return new_solution

  def get_name(self):
    return "FeatureSelectionProblem"
  
#DATA
df_hd = pd.read_csv('D:/Doctorado/doctorado/Data/HD_filtered.csv')

#PRE-SETS
encoder = LabelEncoder()
X = df_hd.drop(columns=['Samples','Grade']).to_numpy()
y = encoder.fit_transform(df_hd.Grade.to_numpy())
clases = list(df_hd.columns[:-2])

#PROBLEM
alfa = 0.9
problem = FeatureSelectionProblem(X,y,alfa)

#PARAMETERS
pobl = 400
off_pobl = int(pobl*0.75)
#mating_size = int(pobl*0.25)
evals = 7500
mut_p = 0.01
cross_p = 0.9

#OPERATORS
mut = BitFlipMutation(mut_p)
cross = SPXCrossover(cross_p)
selection = BestSolutionSelection()
criterion = StoppingByEvaluations(max_evaluations=evals)

# # ALGORITHM
algorithm = CellularGeneticAlgorithm(
    problem=problem,
    population_size=pobl,
    offspring_population_size=off_pobl,
    #mating_pool_size = mating_size,
    mutation=mut,
    crossover=cross,
    selection=selection,
    termination_criterion=criterion
)

algorithm.run()

# # #RESULTS
soluciones_ls = algorithm.get_result()
objectives = soluciones_ls.objectives
variables = soluciones_ls.variables

var_squeezed = np.squeeze(variables)
genes_selected = [gen for gen,var in zip(clases,var_squeezed) if var==1]

with open('Resultados_CGA/Resultados_func_agregativa/Experimento1/Resultados_FS_3.txt','w') as f:
  f.write(f"Name: {algorithm.get_name()}\n")
  f.write(f"Solucion objectives: {objectives}\n")
  f.write(f"Solucion variables: {variables}\n")
  f.write(f"Solucion variables type: {type(variables)}\n")
  f.write(f"Solucion variables amount: {len(variables)}\n")
  f.write(f"Selected genes: {genes_selected}\n")
  f.write(f"Selected genes amount: {len(genes_selected)}\n")
  f.write(f"Agregative function parameters alfa: {alfa}, beta: {1-alfa}\n")
  f.write(f"Stopping criterion: {criterion} | parameter: {evals}\n")
  f.write(f"Crossover: {cross} | parameter: {cross_p}\n")
  f.write(f"Mutation: {mut} | parameter: {mut_p}\n")
  f.write(f"Poblation size: {pobl} | Offspring size: {off_pobl}\n")
  f.write(f"Clases: {encoder.classes_}\n")
  f.write(f"Data input shape: {X.shape}\n")
