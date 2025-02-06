from jmetal.core.algorithm import Algorithm
from jmetal.core.problem import Problem
from jmetal.config import store
from jmetal.util.termination_criterion import TerminationCriterion
from jmetal.core.solution import BinarySolution
from jmetal.util.generator import Generator
from jmetal.util.evaluator import Evaluator
from jmetal.util.comparator import Comparator

from typing import List, TypeVar
import numpy as np
import time
import threading

S = TypeVar("S")
R = TypeVar("R")

class Ant:
    def __init__(self, problem: Problem):
        self.problem = problem
        self.solution = problem.create_solution()
        self.pheromone_trail = [0.5] * self.problem.number_of_variables

    def construct_solution(self, pheromone_trail, heuristic_info, alpha, beta):
        for i in range(self.problem.number_of_variables):
            probability = (pheromone_trail[i] ** alpha) * (heuristic_info[i] ** beta)
            if np.random.rand() < probability:
                self.solution.variables[i] = True
            else:
                self.solution.variables[i] = False
        self.problem.evaluate(self.solution)

class BinaryACO(Algorithm[S, R], threading.Thread):
    def __init__(self, 
                problem:Problem, 
                colony_size: int,
                alpha: float, 
                beta: float, 
                evaporation_rate: float,
                termination_criterion: TerminationCriterion = store.default_termination_criteria,
                population_generator: Generator = store.default_generator,
                population_evaluator: Evaluator = store.default_evaluator,
                dominance_comparator: Comparator = store.default_comparator):
        super(BinaryACO, self).__init__()
        self.problem = problem
        self.colony_size = colony_size
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate

        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)

        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.dominance_comparator = dominance_comparator

        self.ants = [Ant(problem) for _ in range(colony_size)]
        self.pheromone_trail = [1.0/self.problem.number_of_variables]*self.problem.number_of_variables
        self.heuristic_info = [1.0/self.problem.number_of_variables]*self.problem.number_of_variables
        self.best_solution = None
        self.best_solution_fitness = float('inf')
        self.solutions = []
        self.evaluations = 0

    def create_initial_solutions(self) -> []:
        return [self.problem.create_solution() for _ in range(self.colony_size)]

    def evaluate(self, solution_list: []) -> []:
        # print(solution_list)
        return self.population_evaluator.evaluate(solution_list, self.problem)

    def init_progress(self) -> None:
        self.best_solution = min(self.solutions, key=lambda s: s.objectives[0])
        # self.best_fitness = self.best_solution.objectives[0]

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def step(self):
        for ant in self.ants:
            ant.construct_solution(self.pheromone_trail, self.heuristic_info, self.alpha, self.beta)
        self.solutions = self.evaluate([ant.solution for ant in self.ants])
        self.update_pheromone_trail()
        self.local_search()
        self.update_best_solution()

    def update_pheromone_trail(self):
        self.pheromone_trail = [max(0.1, min(0.9, pheromone * (1 - self.evaporation_rate))) for pheromone in self.pheromone_trail]
        for ant in self.ants:
            for i in range(self.problem.number_of_variables):
                if ant.solution.variables[i]:
                    objective_value = ant.solution.objectives[0]
                    if objective_value == 0:
                        objective_value = 1e-6
                    self.pheromone_trail[i] = min(0.9, self.pheromone_trail[i] + self.alpha * (1 / objective_value))

    def update_best_solution(self):
        current_best = min(self.solutions, key=lambda s: s.objectives[0])
        if self.dominance_comparator.compare(current_best, self.best_solution) < 0:
            self.best_solution = current_best
            # self.best_fitness = current_best.objectives[0]

    def local_search(self):
        for ant in self.ants:
            for i in range(self.problem.number_of_variables):
                ant.solution.variables[i] = not ant.solution.variables[i]
                self.problem.evaluate(ant.solution)
                if ant.solution.objectives[0] > ant.solution.objectives[0]:
                    ant.solution.variables[i] = not ant.solution.variables[i]  # Revert if not improved
                else:
                    break

    def update_progress(self) -> None:
        self.evaluations += self.colony_size
        # print(f"Evaluations: {self.evaluations}")
        observable_data = self.observable_data()
        self.observable.notify_all(**observable_data)

    def observable_data(self) -> dict:
        return {
            "PROBLEM": self.problem,
            "EVALUATIONS": self.evaluations,
            "SOLUTIONS": self.result(),
            "COMPUTING_TIME": time.time() - self.start_computing_time,
        }

    def result(self) -> List[BinarySolution]:
        # print(f"Results: {len(self.best_solution.variables)}")
        return self.best_solution

    def get_name(self) -> str:
        return 'Binary Ant Colony Optimization Algorithm'
