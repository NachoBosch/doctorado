import numpy as np
from jmetal.core.problem import Problem
from jmetal.core.solution import BinarySolution
from jmetal.core.algorithm import Algorithm
from jmetal.util.termination_criterion import TerminationCriterion
from jmetal.util.evaluator import Evaluator
from typing import List
import random

class BinaryBOA(Algorithm[BinarySolution, BinarySolution]):
    def __init__(self, 
                 problem: Problem,
                 population_size: int,
                 a: float,
                 c: float,
                 p: float,
                 termination_criterion: TerminationCriterion,
                 evaluator: Evaluator):
        
        super().__init__()
        self.problem = problem
        self.population_size = population_size
        self.a = a  # Sensory modality constant
        self.c = c  # Power exponent for fragrance
        self.p = p  # Switching probability
        self.termination_criterion = termination_criterion
        self.evaluator = evaluator
        self.population = []
        self.best_solution = None

    def create_initial_solutions(self) -> List[BinarySolution]:
        return [self.problem.create_solution() for _ in range(self.population_size)]
    
    def calculate_fragrance(self, fitness) -> float:
        return self.a * (fitness ** self.c)
    
    def update_position(self, butterfly: BinarySolution, best_solution: BinarySolution, fragrance: float):
        for i in range(len(butterfly.variables)):
            if random.random() < fragrance:
                butterfly.variables[i] = best_solution.variables[i]
            else:
                butterfly.variables[i] = not butterfly.variables[i] if random.random() < 0.5 else butterfly.variables[i]

    def evolve_population(self):
        for butterfly in self.population:
            fragrance = self.calculate_fragrance(butterfly.objectives[0])
            if random.random() < self.p:
                self.update_position(butterfly, self.best_solution, fragrance)
            else:
                random_butterfly = random.choice(self.population)
                self.update_position(butterfly, random_butterfly, fragrance)

    def init_progress(self):
        self.population = self.create_initial_solutions()
        self.evaluator.evaluate(self.population, self.problem)
        self.best_solution = min(self.population, key=lambda sol: sol.objectives[0])
        
    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def step(self):
        self.evolve_population()
        self.evaluator.evaluate(self.population, self.problem)
        current_best = min(self.population, key=lambda sol: sol.objectives[0])
        if current_best.objectives[0] < self.best_solution.objectives[0]:
            self.best_solution = current_best

    def result(self) -> BinarySolution:
        return self.best_solution

    def get_name(self) -> str:
        return "Binary Butterfly Optimization Algorithm"

