from jmetal.problem.singleobjective.unconstrained import Sphere
from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.operator.mutation import PolynomialMutation
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.selection import BestSolutionSelection
from jmetal.util.termination_criterion import StoppingByEvaluations

from jmetal.util.observer import ProgressBarObserver
from jmetal.util.solution import get_non_dominated_solutions
from jmetal.lab.visualization import Plot

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

pbar = ProgressBarObserver(max=100)

problema = Sphere(number_of_variables=10)

algoritmo_ga = GeneticAlgorithm(problema,
                            population_size=100,
                            offspring_population_size=20,
                            mutation=PolynomialMutation(1.0 / problema.number_of_variables, 20.0), 
                            crossover=SBXCrossover(0.9, 20.0),
                            selection=BestSolutionSelection(),
                            termination_criterion=StoppingByEvaluations(max_evaluations=100)
                            )

algoritmo_ga.observable.register(observer=pbar)
algoritmo_ga.run()
soluciones_ga = algoritmo_ga.get_result()
print(f"Soluciones Genetic Algorithm: {soluciones_ga.variables}\n")
print(f"Mejor solución: {soluciones_ga.objectives}\n")