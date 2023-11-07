from jmetal.problem.singleobjective.unconstrained import OneMax
from jmetal.core.problem import BinaryProblem
from jmetal.core.solution import BinarySolution
from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.operator.mutation import BitFlipMutation
from jmetal.operator.crossover import CXCrossover
from jmetal.operator.selection import BestSolutionSelection
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.algorithm.singleobjective.local_search import LocalSearch
import matplotlib.pyplot as plt 


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
problema = OneMax(number_of_bits=100)

#definimos el algoritmo
algoritmo_ga = GeneticAlgorithm(problema,
                            population_size=50,
                            offspring_population_size=20,
                            mutation=BitFlipMutation(0.1), # probabilidad de mutación del 0.1 para cada bit, de que sea invertido en cada generación (0->1, 1->0)
                            crossover=CXCrossover(0.1), # probabilidad de cruce del 0.9. de que para cada par de soluciones en la población se realice el cruce y así generar una nueva solución en cada generación
                            selection=BestSolutionSelection(),
                            termination_criterion=StoppingByEvaluations(max_evaluations=50)
                            )

#corremos el algoritmo
# algoritmo_ga.run()

#obtenemos las soluciones a las que llegó el algoritmo
# soluciones_ga = algoritmo_ga.get_result()
# mejor_solucion = soluciones_ga[0] 

# print(f"Type: {type(soluciones_ga)}")
# print(f"Soluciones Genetic Algorithm: {soluciones_ga.variables}\n")
# print(f"Mejor solución: {soluciones_ga.objectives}\n")
# print(f"Cadena de bits: {len(mejor_solucion.variables)} | {mejor_solucion.variables}")


evolucion = []
mejor_solucion = 0 
for generacion in range(100):
    algoritmo_ga.run()
    soluciones = algoritmo_ga.get_result()
    mejor_solucion_generacion = soluciones.objectives[0]

    evolucion.append(mejor_solucion_generacion)

    if mejor_solucion_generacion <= min(evolucion):
        mejor_solucion = mejor_solucion_generacion

print(f"Mejor solución encontrada: {mejor_solucion}")

plt.plot(x=[i for i in range(100)],y= evolucion,c='green')
plt.xlabel("Generación")
plt.ylabel("Suma")
plt.title("Suma de bits en One Max")
plt.show()



# LOCAL SEARCH 
algoritmo_ls = LocalSearch(problema,
                            mutation=BitFlipMutation(0.5),
                            termination_criterion=StoppingByEvaluations(max_evaluations=50)
                            )

algoritmo_ls.run()
soluciones_ls = algoritmo_ls.get_result()
print(f"Local Search: {soluciones_ls}")