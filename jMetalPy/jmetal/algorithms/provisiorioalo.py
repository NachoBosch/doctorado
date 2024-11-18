import numpy as np
from jmetal.core.problem import BinaryProblem
from jmetal.core.solution import BinarySolution
from jmetal.core.algorithm import Algorithm
from typing import List, TypeVar

S = TypeVar('S', bound=BinarySolution)

class BinaryAntLionOptimizer(Algorithm[S, List[S]]):
    def __init__(self,
                 problem: BinaryProblem,
                 n_ants: int,
                 n_antlions: int,
                 max_iterations: int):
        super().__init__()
        self.problem = problem
        self.n_ants = n_ants
        self.n_antlions = n_antlions
        self.max_iterations = max_iterations
        self.elite = None
        
    def roulette_wheel_selection(self, fitness_values):
        # Normalizar fitness para probabilidades positivas
        fitness_prob = fitness_values - np.min(fitness_values)
        if np.sum(fitness_prob) == 0:
            return np.random.randint(0, len(fitness_values))
        else:
            fitness_prob = fitness_prob / np.sum(fitness_prob)
            return np.random.choice(len(fitness_values), p=fitness_prob)
    
    def binary_random_walk(self, current_position: List[bool], mutation_rate: float) -> List[bool]:
        """
        Implementa el random walk para representación binaria
        """
        new_position = current_position.copy()
        for i in range(len(new_position)):
            if np.random.random() < mutation_rate:
                new_position[i] = not new_position[i]
        return new_position
    
    def binary_crossover(self, parent1: List[bool], parent2: List[bool]) -> List[bool]:
        """
        Implementa crossover uniforme para representación binaria
        """
        child = []
        for bit1, bit2 in zip(parent1, parent2):
            if np.random.random() < 0.5:
                child.append(bit1)
            else:
                child.append(bit2)
        return child
    
    def calculate_mutation_rate(self, current_iteration):
        """
        Calcula la tasa de mutación basada en la iteración actual
        """
        return 0.1 + 0.9 * (1 - current_iteration/self.max_iterations)
    
    def run(self) -> List[S]:
        # Inicializar poblaciones
        ants = [self.problem.create_solution() for _ in range(self.n_ants)]
        antlions = [self.problem.create_solution() for _ in range(self.n_antlions)]
        
        # Evaluar soluciones iniciales
        for ant in ants:
            self.problem.evaluate(ant)
        for antlion in antlions:
            self.problem.evaluate(antlion)
            
        self.elite = min(antlions, key=lambda x: x.objectives[0])
        
        # Bucle principal
        iteration = 0
        while iteration < self.max_iterations:
            mutation_rate = self.calculate_mutation_rate(iteration)
            
            # Para cada hormiga
            for i, ant in enumerate(ants):
                # Seleccionar antlion usando ruleta
                antlion_idx = self.roulette_wheel_selection(
                    [a.objectives[0] for a in antlions]
                )
                selected_antlion = antlions[antlion_idx]
                
                # Realizar random walks
                rw1 = self.binary_random_walk(
                    selected_antlion.variables[0], 
                    mutation_rate
                )
                rw2 = self.binary_random_walk(
                    self.elite.variables[0], 
                    mutation_rate
                )
                
                # Realizar crossover
                new_position = self.binary_crossover(rw1, rw2)
                
                # Actualizar posición de la hormiga
                ant.variables[0] = new_position
                self.problem.evaluate(ant)
                
                # Actualizar antlion si la hormiga es mejor
                if ant.objectives[0] < antlions[antlion_idx].objectives[0]:
                    antlions[antlion_idx] = ant.copy()
                    
                    # Actualizar elite si es necesario
                    if ant.objectives[0] < self.elite.objectives[0]:
                        self.elite = ant.copy()
            
            iteration += 1
        
        return antlions

    def get_result(self) -> List[S]:
        return [self.elite]

    def get_name(self) -> str:
        return "Binary Ant Lion Optimizer"