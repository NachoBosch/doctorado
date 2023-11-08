from jmetal.problem.singleobjective.tsp import TSP
from jmetal.util.observer import ProgressBarObserver
from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.operator.mutation import BitFlipMutation
from jmetal.operator.crossover import CXCrossover
from jmetal.operator.selection import BestSolutionSelection
from jmetal.util.termination_criterion import StoppingByEvaluations

pbar = ProgressBarObserver(max=100)

problema = TSP('../Data/usacities.tsp')
problema.observable.register(observer=pbar)

#definimos el algoritmo
algoritmo_ga = GeneticAlgorithm(problema,
                            population_size=100,
                            offspring_population_size=20,
                            mutation=BitFlipMutation(1.0 / problema.number_of_variables), # probabilidad de mutación del 0.1 para cada bit, de que sea invertido en cada generación (0->1, 1->0)
                            crossover=CXCrossover(0.5), # probabilidad de cruce del 0.9. de que para cada par de soluciones en la población se realice el cruce y así generar una nueva solución en cada generación
                            selection=BestSolutionSelection(),
                            termination_criterion=StoppingByEvaluations(max_evaluations=100)
                            )

#corremos el algoritmo
algoritmo_ga.run()

#obtenemos las soluciones a las que llegó el algoritmo
soluciones_ga = algoritmo_ga.get_result()
print(f"Soluciones Genetic Algorithm: {soluciones_ga.variables}\n")
print(f"Mejor solución: {soluciones_ga.objectives}\n")