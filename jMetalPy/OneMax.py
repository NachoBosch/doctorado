from jmetal.problem.singleobjective.unconstrained import OneMax
from jmetal.core.problem import BinaryProblem
from jmetal.core.solution import BinarySolution
from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.operator.mutation import BitFlipMutation
from jmetal.operator.crossover import CXCrossover
from jmetal.operator.selection import BestSolutionSelection
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.algorithm.singleobjective.local_search import LocalSearch


# class OneMax(BinaryProblem):
#     def __init__(self,n_bits:int):
#         super(OneMax, self).__init__()
#         self.number_of_variables = n_bits
#         self.number_of_objectives = 1
#         # self.number_of_bits = n_bits
    
#     def create_solution(self) -> BinarySolution:
#         return BinarySolution(self.number_of_variables,
#                               self.number_of_objectives)
#                             #   self.number_of_bits)
    
#     def evaluate(self,solution) -> BinarySolution:
#         sum_bits = sum(solution.variables[0])
#         solution.objectives[0] = sum_bits
#         return solution
    
#     def get_name(self):
#         return "OneMax"
    
#definimos el problema
problema = OneMax(number_of_bits=40)

#definimos el algoritmo
algoritmo_ga = GeneticAlgorithm(problema,
                            population_size=40,
                            offspring_population_size=40,
                            mutation=BitFlipMutation(0.1),
                            crossover=CXCrossover(0.9),
                            selection=BestSolutionSelection(),
                            termination_criterion=StoppingByEvaluations(max=100)
                            )

#corremos el algoritmo
algoritmo_ga.run()

#obtenemos las soluciones a las que llegó el algoritmo
soluciones_ga = algoritmo_ga.get_result()
print(f"Soluciones Genetic Algorithm: {soluciones_ga}")

#Local Search
# algoritmo_ls = LocalSearch(problema,
#                             mutation=BitFlipMutation(0.1),
#                             termination_criterion=StoppingByEvaluations(max=40)
#                             )