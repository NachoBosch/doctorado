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
import copy
import time

class BinaryHHO(Algorithm[BinarySolution, BinarySolution]):
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
        self.population = []
        self.best_solution = None
        self.evaluations = 0

    def create_initial_solutions(self) -> List[BinarySolution]:
        return [self.problem.create_solution() for _ in range(self.population_size)]

    def evaluate(self, solution_list: List[BinarySolution]):
        return self.population_evaluator.evaluate(solution_list, self.problem)

    def init_progress(self):
        self.population = self.create_initial_solutions()
        self.evaluate(self.population)
        self.best_solution = min(self.population, key=lambda sol: sol.objectives[0])
        self.evaluations = self.population_size

    def s_shape_transfer(self, x):
        """Función de transferencia S-shape."""
        # return 1 / (1 + np.exp(-x))
        return abs(np.tanh(x))

    def update_binary_position(self, probability):
        return True if random.random() < probability else False

    def hawk_position_update(self, hawk, rabbit, escape_energy):
        """Actualiza la posición de un halcón según su cercanía al conejo (óptimo local)."""
        updated_hawk = copy.deepcopy(hawk)

        for i in range(self.problem.number_of_variables):
            if abs(escape_energy) >= 1:
                # Exploración: alejarse del conejo
                q = random.random()
                if q >= 0.5:
                    new_velocity = rabbit.variables[i]  # Cerca del conejo
                else:
                    rand_hawk = random.choice(self.population)  # Movimiento aleatorio en la población
                    new_velocity = rand_hawk.variables[i]
            else:
                # Explotación: acercarse al conejo
                r = random.random()
                if r >= 0.5:
                    # Movimiento directo hacia el conejo
                    new_velocity = rabbit.variables[i]
                else:
                    # Actualización de la posición basada en espiral
                    dist = np.random.normal() * (rabbit.variables[i] - hawk.variables[i])
                    new_velocity = hawk.variables[i] + escape_energy * dist

            # Aplicar la función de transferencia para convertir a binario
            prob = self.s_shape_transfer(new_velocity)
            updated_hawk.variables[i] = self.update_binary_position(prob)

        return updated_hawk

    def update_positions(self, rabbit):
        """Actualiza la posición de cada halcón."""
        escape_energy = 2 * (1 - self.evaluations / self.population_size)  # Energía de escape
        for idx, hawk in enumerate(self.population):
            # Actualización de posición
            updated_hawk = self.hawk_position_update(hawk, rabbit, escape_energy)
            self.population[idx] = updated_hawk

    def update_best_solution(self):
        """Actualiza la mejor solución (conejo) encontrada hasta ahora."""
        current_best = min(self.population, key=lambda sol: sol.objectives[0])
        if self.dominance_comparator.compare(current_best, self.best_solution) < 0:
            self.best_solution = copy.deepcopy(current_best)

    def step(self):
        rabbit = self.best_solution  # El conejo es la mejor solución hasta ahora
        self.update_positions(rabbit)
        self.evaluate(self.population)
        self.update_best_solution()

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def update_progress(self):
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
        print(f"Result: {self.best_solution.objectives[0]}")
        return self.best_solution

    def get_name(self):
        return "Binary Harris Hawks Optimizer (HHO)"
