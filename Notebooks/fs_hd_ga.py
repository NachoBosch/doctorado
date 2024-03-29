import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np

# from jmetal.core.problem import BinaryProblem
from jmetal.core.solution import BinarySolution
from jmetal.algorithm.singleobjective import GeneticAlgorithm
# from jmetal.algorithm.multiobjective import NSGAII
from jmetal.operator import BinaryTournamentSelection, SBXCrossover, BitFlipMutation, DifferentialEvolutionCrossover, PolynomialMutation, CXCrossover
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.operator.selection import BestSolutionSelection


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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
    X_selected = self.X[:, selected_features]
    Xtrain,Xtest,ytrain,ytest = train_test_split(X_selected,self.y)

    model = DecisionTreeClassifier()
    # model = AdaBoostClassifier(n_estimators=100)
    model.fit(Xtrain, ytrain)
    y_pred = model.predict(Xtest)
    acc = accuracy_score(ytest, y_pred)

    solution.objectives[0] = acc
    solution.constraints = []

  def create_solution(self):
    new_solution = BinarySolution(
        number_of_variables = self.number_of_variables,
        number_of_objectives = self.number_of_objectives,
        number_of_constraints = self.number_of_constraints
    )
    # new_variables = [list(np.random.randint(0, 2, size=1).tolist()[0] for _ in range(self.number_of_variables))]
    new_variables = [np.random.randint(0, 2, size=1).tolist() for _ in range(self.number_of_variables)]
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
encoder = LabelEncoder()
X = df_hd.drop(columns=['Samples','Grade']).to_numpy()
y = encoder.fit_transform(df_hd.Grade.to_numpy())

problem = FeatureSelectionProblem(X,y)

#ALGORITHM
algorithm = GeneticAlgorithm(
    problem=problem,
    population_size=10,
    offspring_population_size=2,
    mutation=BitFlipMutation(0.1),
    crossover=CXCrossover(0.1),
    selection=BestSolutionSelection(),
    termination_criterion=StoppingByEvaluations(max_evaluations=50)
)

algorithm.run()

#RESULTS
soluciones_ls = algorithm.get_result()
objectives = soluciones_ls.objectives
variables = soluciones_ls.variables

with open('resultados.txt','w') as f:
  # f.write(f"Target encoded: {y}\n")
  f.write(f"Solucion objectives: {objectives}\n")
  f.write(f"Solucion variables: {variables}\n")
  f.write(f"Solucion variables type: {type(variables)}\n")
  f.write(f"Solucion variables amount: {len(variables)}\n")
