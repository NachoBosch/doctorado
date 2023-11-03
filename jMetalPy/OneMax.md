# **One Max optimization problem**

El problema se reduce a una pregunta: ¿cuál es la suma máxima de una cadena de bits (una cadena que consta de solo 1s y 0s) de longitud N?

Por supuesto, se entiende que la suma máxima de una cadena de bits de longitud N es igual a N. Sin embargo, si se quisiera probar esto utilizando fuerza bruta, terminaría con una busqueda de 2^N diferentes soluciones.

Este problema `One Max` de optimización combinatoria simple es utilizado como problema de prueba (*benchmark*) para evaluar y comparar algoritmos de búsqueda y optimización.


## JMetalPy

* **classjmetal.problem.singleobjective.unconstrained.OneMax(number_of_bits: int = 256, rf_path: str = None)**
* *Bases: jmetal.core.problem.BinaryProblem*
    * create_solution() → jmetal.core.solution.BinarySolution
    * evaluate(solution: jmetal.core.solution.BinarySolution) → jmetal.core.solution.BinarySolution

* classjmetal.core.problem.BinaryProblem(rf_path: str = None)
    * create_solution() → jmetal.core.solution.BinarySolution
    * evaluate(solution: jmetal.core.solution.BinarySolution) → jmetal.core.solution.BinarySolution

* classjmetal.core.solution.BinarySolution(number_of_variables: int, number_of_objectives: int, number_of_constraints: int = 0)
    * get_total_number_of_bits() → int

* **Genetic Algorithm**
* class jmetal.algorithm.singleobjective.genetic_algorithm.GeneticAlgorithm(problem: jmetal.core.problem.Problem, population_size: int, offspring_population_size: int, mutation: jmetal.core.operator.Mutation, crossover: jmetal.core.operator.Crossover, selection: jmetal.core.operator.Selection, termination_criterion: jmetal.util.termination_criterion.TerminationCriterion = <jmetal.util.termination_criterion.StoppingByEvaluations object>, population_generator: jmetal.util.generator.Generator = <jmetal.util.generator.RandomGenerator object>, population_evaluator: jmetal.util.evaluator.Evaluator = <jmetal.util.evaluator.SequentialEvaluator object>)
