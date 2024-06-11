# Modulos necesarios de jMetalPy
from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.algorithm.singleobjective.evolution_strategy import EvolutionStrategy
from jmetal.operator import SBXCrossover, PolynomialMutation, BinaryTournamentSelection
from jmetal.core.problem import FloatProblem
from jmetal.util.solution import FloatSolution
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.solution import get_non_dominated_solutions
from jmetal.lab.visualization import Plot
from random import uniform

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn import metrics as ms

# Cargamos dataset de prueba
data = pd.read_csv('../Data/MSFT.csv')
print(f"Observamos algunos registros:\n {data.head(2)}")

print(f"Observamos info del dataset:\n {data.info()}")

plt.figure()
data['Close'].plot(figsize=(10,7))
plt.show()

print(f"Correlación de las variables:\n{data.corr()}")

X = data[['Open','High']].to_numpy()
y = data['Close'].to_numpy()
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

class SVRprueba1:
  def __init__(self, Xtrain, ytrain, Xtest, ytest):
    self.number_of_variables = 2  # Hyperparametro C y epsilon
    self.number_of_objectives = 1  # Mean absolute error
    self.number_of_constraints = 0
    self.lower_bound = [1,0.5]
    self.upper_bound = [5,1.0]
    self.Xtrain = Xtrain
    self.Xtest = Xtest
    self.ytrain = ytrain
    self.ytest = ytest

  def evaluate(self, solution: FloatSolution) -> FloatSolution:
    c = solution.variables[0]
    e = solution.variables[1]
    svr = SVR(kernel='linear',C=c ,epsilon=e)
    svr.fit(self.Xtrain, self.ytrain.ravel())
    y_predict = svr.predict(self.Xtest)
    solution.objectives[0] = ms.mean_absolute_error(self.ytest, y_predict)
    return solution

  def create_solution(self) -> FloatSolution:
    lower_bound = [self.lower_bound[0],self.lower_bound[1]]
    upper_bound = [self.upper_bound[0],self.upper_bound[1]]
    solution = FloatSolution(
        lower_bound, upper_bound, self.number_of_objectives, self.number_of_constraints
    )
    solution.variables[0] = np.random.uniform(self.lower_bound[0], self.upper_bound[0])
    solution.variables[1] = np.random.uniform(self.lower_bound[1], self.upper_bound[1])
    return solution

  def get_name(self):
    return "SVRprueba1"

#Instanciamos la clase SVRprueba1
svr_problem = SVRprueba1(Xtrain, ytrain, Xtest, ytest)

# Seteamos los hiperparametros del algoritmo
algorithm = GeneticAlgorithm(
    problem=svr_problem,
    population_size=100,
    offspring_population_size=100,
    selection = BinaryTournamentSelection(),
    mutation=PolynomialMutation(probability=1.0 / svr_problem.number_of_variables, distribution_index=20),
    crossover=SBXCrossover(probability=1.0, distribution_index=20),
    termination_criterion=StoppingByEvaluations(max=25000)
)

# Corremos la optimización
algorithm.run()

# Una vez terminado, obtenemos los resultados con el método get_result()
solution = algorithm.get_result()
print(f"C solución: {solution.variables[0]}\nEpsilon solución:{solution.variables[1]}")

# Almacenamos los parámetros
with open('GeneticAlgorithmOptimizer.txt','w') as f:
  file = f.read()
  file.write(solution.variables[0])
  file.write(f"\n{solution.variables[1]}")
  file.close()

#Ploteamos frontera de Pareto
front = get_non_dominated_solutions(algorithm.get_result())
plot_front = Plot(title='Pareto front approximation', axis_labels=['x', 'y'])
plot_front.plot(front, label='SVR', filename='svr', format='png')