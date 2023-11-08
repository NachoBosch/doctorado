from jmetal.problem.singleobjective.unconstrained import SubsetSum
from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.operator.mutation import IntegerPolynomialMutation
from jmetal.operator.crossover import PMXCrossover
from jmetal.operator.selection import BestSolutionSelection
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.observer import ProgressBarObserver

pbar = ProgressBarObserver(max=100)

pesos = [10, 20, 15, 25, 5]
objetivo = 30
problema = SubsetSum(C = objetivo,
                     W = pesos)


algoritmo_ga = GeneticAlgorithm(problema,
                            population_size=100,
                            offspring_population_size=20,
                            mutation=IntegerPolynomialMutation(0.1, 20), 
                            crossover=PMXCrossover(0.6), # probabilidad de cruce del 0.9. de que para cada par de soluciones en la población se realice el cruce y así generar una nueva solución en cada generación
                            selection=BestSolutionSelection(),
                            termination_criterion=StoppingByEvaluations(max_evaluations=100)
                            )

algoritmo_ga.observable.register(observer=pbar)

#corremos el algoritmo
algoritmo_ga.run()

#obtenemos las soluciones a las que llegó el algoritmo
soluciones_ga = algoritmo_ga.get_result()
print(f"Soluciones Genetic Algorithm: {soluciones_ga.variables}\n")
print(f"Mejor solución: {soluciones_ga.objectives}\n")
