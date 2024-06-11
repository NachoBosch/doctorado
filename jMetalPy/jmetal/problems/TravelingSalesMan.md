# **TSP**

Problema de optimización combinatoria discreta. 

Se busca la ruta más corta que pase por un conjunto de ciudades y regrese al punto de partida, visitando una vez cada ciudad.

Es un problema NP-hard (no deterministic polynomial hard) es decir, posee una dificultad alta computacionalmente para ser resuelto de manera eficiente. 

Es dificultoso debido a que es NP, es decir, no toma un tiempo polinómico encontrar la solución. 

## JMetalPy

* **class jmetal.problem.singleobjective.tsp.TSP(instance: str = None)**
* *Bases: jmetal.core.problem.PermutationProblem*
    * create_solution() → jmetal.core.solution.PermutationSolution
    `Creates a random_search solution to the problem.`
    `Returns -> Solution`
    * evaluate(solution: jmetal.core.solution.PermutationSolution) → jmetal.core.solution.PermutationSolution
    `Evaluate a solution. For any new problem inheriting from Problem, this method should be replaced. Note that this framework ASSUMES minimization, thus solutions must be evaluated in consequence.`
    `Returns -> Evaluated solution`
    * get_name()
* property: number_of_cities
