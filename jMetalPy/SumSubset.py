from jmetal.problem.singleobjective.unconstrained import SubsetSum
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

pbar = ProgressBarObserver(max=100)

pesos = [10, 20, 15, 25, 5]
objetivo = 30

'''
Problema de optimización combinatoria que busca un subconjunto de elementos, dado un conjunto, de tal manera que la suma de los elementos en el subconjunto sea igual a un valor objetivo específico.
Tiene dos atributos importantes que son: C (large integer) valor objetivo y W (set of non-negative integers) array que representa el conjunto  de elementos.
'''
problema = SubsetSum(C = objetivo,
                     W = pesos)


algoritmo_ga = GeneticAlgorithm(problema,
                            population_size=100,
                            offspring_population_size=20,
                            mutation=IntegerPolynomialMutation(0.1, 20), 
                            crossover=PMXCrossover(0.6),
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


## Optmización con gráfica
evolucion = []
mejor_solucion = 0 
for generacion in range(100):
    algoritmo_ga.run()
    soluciones = algoritmo_ga.get_result()
    mejor_solucion_generacion = soluciones.objectives[0]
    evolucion.append(mejor_solucion_generacion)
    if mejor_solucion_generacion <= min(evolucion):
        mejor_solucion = mejor_solucion_generacion

print(f"Mejor solución: {mejor_solucion}")
plt.plot(evolucion,c='purple')
plt.xlabel("Generación")
plt.ylabel("Suma")
plt.title("Suma de enteros")
plt.show()

# Pareto front
# Esto sirve en problemas del tipo continuos
# front = get_non_dominated_solutions(algoritmo_ga.get_result())
# plot_front = Plot('Aproximación Pareto front')
# plot_front.plot(front,label='SubsetSum',format='png')