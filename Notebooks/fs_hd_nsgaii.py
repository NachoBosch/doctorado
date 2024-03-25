import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# from jmetal.core.problem import BinaryProblem
from jmetal.core.solution import BinarySolution
# from jmetal.algorithm.singleobjective import GeneticAlgorithm
from jmetal.algorithm.multiobjective import NSGAII
from jmetal.operator import BinaryTournamentSelection, SBXCrossover, BitFlipMutation, DifferentialEvolutionCrossover, PolynomialMutation, CXCrossover
from jmetal.util.termination_criterion import StoppingByEvaluations

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#PROBLEM
class FeatureSelectionProblem():
  def __init__(self,X,y):
    self.X = X
    self.y = y
    self.number_of_variables = X.shape[1]
    self.number_of_objectives = 1
    self.number_of_constraints = 0

  def evaluate(self, solution):
    selected_features = np.flatnonzero(solution.variables)
    X_selected = self.X.iloc[:, selected_features]
    Xtrain,Xtest,ytrain,ytest = train_test_split(X_selected,self.y)

    model = DecisionTreeClassifier()
    model.fit(Xtrain, ytrain)
    y_pred = model.predict(Xtest)
    acc = accuracy_score(ytest, y_pred)
    print(f"Accuracy: {acc}")

    solution.objectives[0] = acc
    solution.constraints = []

  def create_solution(self):
    new_solution = BinarySolution(
        number_of_variables = self.number_of_variables,
        number_of_objectives = self.number_of_objectives,
        number_of_constraints = self.number_of_constraints
    )
    # new_variables = [list(np.random.randint(0, 2, size=1).tolist()[0] for _ in range(self.number_of_variables))]
    new_variables = [np.random.randint(0, 2, size=1)[0] for _ in range(self.number_of_variables)]
    new_solution.variables = new_variables
    return new_solution

  def get_name(self):
    return "FeatureSelectionProblem"
  
#DATA
df_hd = pd.read_csv('../Data/HD_dataset_full.csv')
df_hd.rename(columns={'Unnamed: 0':'Samples'},inplace=True)
df_hd['Grade'] = df_hd['Grade'].map({'-':'Control',
                                     '0':'HD_0',
                                     '1':'HD_1',
                                     '2':'HD_2',
                                     '3':'HD_3',
                                     '4':'HD_4'})

#PRE-SETS
X = df_hd.drop(columns=['Samples','Grade'])
y = df_hd.Grade
problem = FeatureSelectionProblem(X,y)

#ALGORITHM
algorithm = NSGAII(
    problem=problem,
    population_size=100,
    offspring_population_size=100,
    mutation=BitFlipMutation(probability=1.0 / problem.number_of_variables),
    crossover=CXCrossover(probability=1.0),
    termination_criterion=StoppingByEvaluations(max_evaluations=500)
)

algorithm.run()

#RESULTS
soluciones_ls = algorithm.get_result()
print(f"NSGAII: {soluciones_ls}")
