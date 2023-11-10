from jmetal.problem.singleobjective.unconstrained import Sphere
from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.operator.mutation import IntegerPolynomialMutation
from jmetal.operator.crossover import PMXCrossover
from jmetal.operator.selection import BestSolutionSelection
from jmetal.util.termination_criterion import StoppingByEvaluations

from jmetal.util.observer import ProgressBarObserver
from jmetal.util.solution import get_non_dominated_solutions
from jmetal.lab.visualization import Plot

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

problema = Sphere(number_of_variables=10)

