# Doctorado

Este repositorio contiene todo lo relacionado a inteligencia computacional aplicada a la investigación de enfermedad de Hungtington

## jMetalPy

En este directorio se encuentra la documentación de los casos de prueba junto a los scripts correspondientes en Python.

Hay ejemplo de problemas de optimización combinatorios y continuos.

## Operadores

1. **CrossOver:**
    
    *   *CXCrossover* (cycle crossover) diseñado para problemas de optimización combinatoria que involucran permutaciones, como el problema TSP o el problema de la asignación cuadrática QAP. Opera mediante la creación de ciclos de elementos en las dos soluciones padre y luego alterna los elementos entre los padres para crear dos soluciones descendientes. Resulta efectivo para preservar la estructura de las permutaciones en problemas donde el orden de los elementos es importante.

    *   *SBX* (simulated binary crossover) sirve para problemas de optimización continuos. Ajustando el índice de distribución y el índice de cruce, se controla la exploración y explotación del espacio de búsqueda, permitiendo un equilibrio entre la diversidad y la convergencia en la búsqueda de soluciones óptimas.
    
    * *PMXCrossover* (partially mapped crossover) utilizado en problemas de optimización combinatoria, especificamente en aquellos con permutaciones. Mediante mezclar parcialmente dos padres crea descendencia, preservando la estructura de los elementos comunes entre ellos. El operador selecciona una sección aleatoria de un padre y la copia directamente a la descendencia, luego completa el resto de la descendencia utilizando información del otro padre, asegurando que no se dupliquen ni falten elementos.  

2. **Mutation:**

    *   *BitFlipMutation* mutación diseñada para problemas de optimización combinatoria (representación binaria) que invierte aleatoriamente los bits. Para etso se usa una probabilidad con la cual será invertido cada bit. Sirve para explorar otras regiones del espacio de búsqueda introduciendo cambios locales en las soluciones.

    *   *IntegerPolynomialMutation* diseñado para problemas de optimización con variables enteras, realiza mutaciones en las variables enteras de una solución mediante mutación polinómica, que implica agregar un valor aleatorio siguiendo una distribución polinómica, a cada variable entera en la solución, así con cambios pequeños pero aleatorios en las variables enteras se introduce variabilidad en las soluciones al realizar.

    *   *PolynomialMutation* utilizado en problemas de optimización continua, realiza mutaciones en las variables continuas de una solución del tipo polinómica. Esta implica agregar un valor aleatorio, siguiendo una distribución polinómica, a cada variable continua en la solución. Este proceso introduce variabilidad en las soluciones al realizar cambios pequeños pero aleatorios en las variables continuas.

3. **Selection**

    *   *BestSolutionSelection* elige la mejor solución de entre un conjunto de soluciones implementando la selección por elitismo, lo que significa que seleccionará la mejor solución sin realizar competicion entre las mismas. La mejor solución se determina generalmente comparando los valores de fitness dependiendo del tipo de problema (maximización o minimización).

    *   *BinaryTournamentSelection* este a diferencia de BestSolutionSelection, realiza una competición directa entre dos soluciones y selecciona la mejor de entre ellas. En un torneo binario, se eligen aleatoriamente dos soluciones de la población actual, y la solución con mejor fitness (max o min) se selecciona como la ganadora del torneo y forma parte de la siguiente generación.