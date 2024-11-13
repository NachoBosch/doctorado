import numpy as np
import random
from jmetal.core.problem import Problem
from jmetal.core.solution import BinarySolution
from jmetal.core.algorithm import Algorithm
from jmetal.config import store
from jmetal.util.termination_criterion import TerminationCriterion
from jmetal.util.comparator import Comparator
from jmetal.util.evaluator import Evaluator
from jmetal.util.generator import Generator
from typing import List
import time
import copy

class BinaryEO(Algorithm[BinarySolution, BinarySolution]):
    def __init__(self, 
                 problem: Problem,
                 population_size: int,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 dominance_comparator: Comparator = store.default_comparator):
        
        super().__init__()
        self.problem = problem
        self.population_size = population_size
        self.termination_criterion = termination_criterion
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.dominance_comparator = dominance_comparator
        self.observable.register(termination_criterion)
        self.pool_size = 5
        self.pool = []
        self.solutions = []
        self.best_solution = None
        self.evaluations = 0

    def create_initial_solutions(self) -> List[BinarySolution]:
        self.population = [self.problem.create_solution() for _ in range(self.population_size)]
        return self.population

    def evaluate(self, solution_list: List[BinarySolution]) -> List[BinarySolution]:
        return [self.problem.evaluate(solution) for solution in solution_list]

    def init_progress(self):
        self.solutions = self.evaluate(self.population)
        self.update_pool()
        self.best_solution = min(self.pool, key=lambda s: s.objectives[0])
        self.evaluations = self.population_size

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def sigmoid_transfer(self, x):
        return 1 / (1 + np.exp(-x))
    
    def tan_transfer(self,x):
        return abs(np.tanh(x))

    def update_binary_position(self, probability):
        return True if random.random() < probability else False

    def update_pool(self):
        self.pool = sorted(self.population, key=lambda s: s.objectives[0])[:self.pool_size]

    def step(self):
        for agent in self.population:
            for i in range(self.problem.number_of_variables):
                equilibrium_point = np.mean([sol.variables[i] for sol in self.pool])
                distance = equilibrium_point - agent.variables[i]
                movement = random.uniform(-1, 1) * distance
                probability = self.tan_transfer(movement)
                agent.variables[i] = self.update_binary_position(probability)
        self.population = self.evaluate(self.population)
        self.update_pool()
        self.update_best_solution()

    def update_best_solution(self):
        current_best = min(self.pool, key=lambda s: s.objectives[0])
        if current_best.objectives[0] < self.best_solution.objectives[0]:
            self.best_solution = copy.deepcopy(current_best)

    def update_progress(self) -> None:
        self.evaluations += self.population_size
        observable_data = self.observable_data()
        self.observable.notify_all(**observable_data)

    def observable_data(self):
        return {
            "PROBLEM": self.problem,
            "EVALUATIONS": self.evaluations,
            "SOLUTIONS": self.result(),
            "COMPUTING_TIME": time.time() - self.start_computing_time,
        }

    def result(self):
        # print(self.best_solution.objectives[0])
        return self.best_solution

    def get_name(self):
        return "Binary Equilibrium Optimizer"
