import numpy as np
from jmetal.core.problem import Problem
from jmetal.core.solution import BinarySolution
from jmetal.core.algorithm import Algorithm
from jmetal.config import store
from jmetal.util.termination_criterion import TerminationCriterion
from jmetal.util.evaluator import Evaluator
from jmetal.util.comparator import Comparator
from jmetal.util.generator import Generator
from typing import List
import random
import time
import copy

class BinaryBOA(Algorithm[BinarySolution, BinarySolution]):
    def __init__(self, 
                 problem: Problem,
                 population_size: int,
                 max_evaluations: int,
                 a: float = 0.5,  # Sensory modality constant
                 c: float = 1.0,  # Power exponent for fragrance
                 p: float = 0.8,  # Switching probability
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 population_evaluator: Evaluator = store.default_evaluator,
                 dominance_comparator: Comparator = store.default_comparator,
                 population_generator: Generator = store.default_generator):
        
        super(BinaryBOA, self).__init__()
        self.problem = problem
        self.population_size = population_size
        self.a = a
        self.c = c
        self.p = p
        self.max_evaluations = max_evaluations
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.population_evaluator = population_evaluator
        self.dominance_comparator = dominance_comparator
        self.population_generator = population_generator
        self.population = []
        self.best_solution = None
        self.evaluations = 0

    def stopping_condition_is_met(self) -> bool:
        if self.termination_criterion:
            return self.termination_criterion.is_met
        return self.evaluations >= self.max_evaluations

    def calculate_fragrance(self, fitness: float) -> float:
        max_fitness = max([s.objectives[0] for s in self.population])
        min_fitness = min([s.objectives[0] for s in self.population])
        
        if max_fitness == min_fitness:
            normalized_fitness = 1.0
        else:
            normalized_fitness = (fitness - min_fitness) / (max_fitness - min_fitness)
        
        # Cálculo adaptativo de la fragancia
        fragrance = self.a * (1 - normalized_fitness) ** self.c
        
        return max(fragrance, 1e-6)  # Asegura un valor mínimo positivo

    def sigmoid(self, x: float) -> float:
        scale = 1.0 + (self.evaluations / self.max_evaluations)  # Factor de escala adaptativo
        # return 1 / (1 + np.exp(-scale * x))
        return 1 / (1 + np.exp(-x))

    def v_shape(self, x: float) -> float:
        return abs(np.tanh(x))

    def update_position(self, butterfly: BinarySolution, reference: BinarySolution, fragrance: float):
        # Probabilidad base usando sigmoide
        prob_s = self.sigmoid(fragrance)
        # Probabilidad complementaria usando v-shape
        prob_v = self.v_shape(fragrance)
        # Factor de diversificación
        diversity_factor = 1.0 - (self.evaluations / self.max_evaluations)
        for i in range(len(butterfly.variables)):
            if random.random() < 0.5:  # Alternar entre S-shape y V-shape
                prob = prob_s
            else:
                prob = prob_v
            if random.random() < prob:
                # Seguir la solución de referencia
                butterfly.variables[i] = reference.variables[i]
            else:
                # Movimiento aleatorio con probabilidad adaptativa
                if random.random() < diversity_factor:
                    butterfly.variables[i] = not butterfly.variables[i]

    def evolve_population(self):
        """Proceso de evolución mejorado con control de diversidad"""
        # Calcular distancia promedio al mejor
        avg_distance = self.calculate_population_diversity()
        
        new_population = []
        for butterfly in self.population:
            new_butterfly = self.problem.create_solution()
            new_butterfly.variables = butterfly.variables.copy()
            
            fragrance = self.calculate_fragrance(butterfly.objectives[0])
            
            # Ajuste adaptativo de la probabilidad global
            adaptive_p = self.p * (1 + avg_distance)
            
            if random.random() < adaptive_p:
                # Búsqueda global
                self.update_position(new_butterfly, self.best_solution, fragrance)
            else:
                # Búsqueda local
                random_butterfly = random.choice(self.population)
                self.update_position(new_butterfly, random_butterfly, fragrance)
            
            new_population.append(new_butterfly)
        
        self.population = new_population

    def calculate_population_diversity(self) -> float:
        if not self.best_solution:
            return 1.0   
        distances = []
        for butterfly in self.population:
            distance = sum(1 for i in range(len(butterfly.variables))
                         if butterfly.variables[i] != self.best_solution.variables[i])
            distances.append(distance / len(butterfly.variables))
        
        return np.mean(distances)

    def create_initial_solutions(self) -> List[BinarySolution]:
        return [self.population_generator.new(self.problem) 
                for _ in range(self.population_size)]

    def init_progress(self):
        self.start_computing_time = time.time()
        self.population = self.create_initial_solutions()
        self.evaluate(self.population)
        self.best_solution = min(self.population, key=lambda sol: sol.objectives[0])
        self.evaluations = self.population_size

    def step(self):
        self.evolve_population()
        self.evaluate(self.population)
        self.update_best_solution()
        self.update_progress()

    def update_best_solution(self):
        current_best = min(self.population, key=lambda sol: sol.objectives[0])
        if current_best.objectives[0] < self.best_solution.objectives[0]:
            self.best_solution = copy.deepcopy(current_best)
            # self.best_solution = self.problem.create_solution()
            # self.best_solution.variables = current_best.variables.copy()
            # self.best_solution.objectives = current_best.objectives.copy()

    def update_progress(self) -> None:
        self.evaluations += self.population_size
        observable_data = self.observable_data()
        self.observable.notify_all(**observable_data)

    def observable_data(self) -> dict:
        return {
            "PROBLEM": self.problem,
            "EVALUATIONS": self.evaluations,
            "SOLUTIONS": self.result(),
            "COMPUTING_TIME": time.time() - self.start_computing_time,
        }
    
    def result(self) -> BinarySolution:
        # print(self.best_solution.objectives[0])
        return self.best_solution
    
    def evaluate(self, solution_list: List[BinarySolution]) -> List[BinarySolution]:
        return self.population_evaluator.evaluate(solution_list, self.problem)

    def get_name(self) -> str:
        return "Binary Butterfly Optimization Algorithm"