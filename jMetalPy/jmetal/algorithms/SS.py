import random
import copy
import time
from typing import List, TypeVar

from jmetal.config import store
from jmetal.core.algorithm import Algorithm
from jmetal.core.operator import Crossover, Mutation
from jmetal.core.problem import Problem
from jmetal.core.solution import Solution
from jmetal.util.comparator import Comparator
from jmetal.util.termination_criterion import TerminationCriterion

S = TypeVar("S")
R = TypeVar("R")

class ScatterSearch(Algorithm[S, R]):
    def __init__(
        self,
        problem: Problem,
        population_size,
        reference_set_size,
        mutation,
        crossover,
        termination_criterion: TerminationCriterion = store.default_termination_criteria,
        comparator: Comparator = store.default_comparator,

    ):
        super(ScatterSearch, self).__init__()
        self.problem = problem
        self.mutation = mutation
        self.crossover = crossover
        self.termination_criterion = termination_criterion
        self.comparator = comparator
        self.population_size = population_size
        self.reference_set_size = reference_set_size
        self.population = []
        self.reference_set = []
        self.observable.register(termination_criterion)

    def create_initial_solutions(self) -> []:
        self.population = [self.problem.create_solution() for _ in range(self.population_size)]
        return self.population

    def evaluate(self, solutions: []) -> []:
        return [self.problem.evaluate(solution) for solution in solutions]

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def init_progress(self) -> None:
        self.evaluations = 0
        self.reference_set = []

    def step(self) -> None:
        if not self.population:
            self.population = self.create_initial_solutions()
            self.evaluate(self.population)
        if not self.reference_set:
            self.reference_set = self.select_reference_set()
        combined_solutions = self.combine_solutions(self.reference_set)
        improved_solutions = self.improve_solutions(combined_solutions)
        self.update_reference_set(improved_solutions)

    def select_reference_set(self) -> []:
        sorted_population = sorted(self.population, key=lambda s: s.objectives[0])
        best_solutions = sorted_population[:self.reference_set_size // 2]
        diverse_solutions = random.sample(sorted_population[self.reference_set_size // 2:], self.reference_set_size // 2)
        return best_solutions + diverse_solutions

    def combine_solutions(self, reference_set: []) -> []:
        new_solutions = []
        for i in range(len(reference_set)):
            for j in range(i + 1, len(reference_set)):
                parents = [reference_set[i], reference_set[j]]
                children = self.crossover.execute(parents)
                child = self.binary_tournament(children)
                new_solutions.append(child)
        return new_solutions
    
    def binary_tournament(self, solutions: List[S]) -> S:
        if len(solutions) != 2:
            raise ValueError(f"Binary tournament requires 2 solutions | {len(solutions)} were given")
        if self.comparator.compare(solutions[0], solutions[1]) == -1:
            return solutions[0]
        else:
            return solutions[1]

    def improve_solutions(self, solutions: []) -> []:
        return [self.mutation.execute(solution) for solution in solutions]

    def update_reference_set(self, solutions: []) -> None:
        for solution in solutions:
            worst_solution = max(self.reference_set, key=lambda s: s.objectives[0])
            if self.comparator.compare(solution, worst_solution) == -1:
                self.reference_set.remove(worst_solution)
                self.reference_set.append(solution)

    def update_progress(self) -> None:
        self.evaluations += len(self.population)
        # self.termination_criterion.update(self)
        observable_data = self.observable_data()
        self.observable.notify_all(**observable_data)
        # print(self.evaluations)

    def observable_data(self) -> dict:
        ctime = time.time() - self.start_computing_time
        return {
            "PROBLEM": self.problem,
            "EVALUATIONS": self.evaluations,
            "SOLUTIONS": self.result(),
            "COMPUTING_TIME": ctime,
        }

    def result(self) -> R:
        return min(self.reference_set, key=lambda s: s.objectives[0])

    def get_name(self) -> str:
        return "Scatter Search"
