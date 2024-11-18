import copy
import time
import logging
import threading
from abc import ABC
from typing import List, TypeVar
import numpy as np
from jmetal.config import store
from jmetal.core.algorithm import Algorithm
from jmetal.core.problem import BinaryProblem
from jmetal.core.solution import BinarySolution
from jmetal.util.evaluator import Evaluator
from jmetal.util.generator import Generator
from jmetal.util.comparator import Comparator
from jmetal.util.termination_criterion import TerminationCriterion

logger = logging.getLogger(__name__)
S = TypeVar('S', bound=BinarySolution)

class BinaryALO(Algorithm[S, List[S]]):
    def __init__(self,
                 problem: BinaryProblem,
                 n_ants: int,
                 n_antlions: int,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 dominance_comparator: Comparator = store.default_comparator):
        super().__init__()
        self.problem = problem
        self.n_ants = n_ants
        self.n_antlions = n_antlions
        self.termination_criterion = termination_criterion
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.dominance_comparator = dominance_comparator
        self.observable.register(termination_criterion)
        
        self.ants = []
        self.antlions = []
        self.elite = None
        self.evaluations = 0

    def create_initial_solutions(self) -> List[S]:
        # Crear poblaciones iniciales de hormigas y ant lions
        self.ants = [self.population_generator.new(self.problem) 
                    for _ in range(self.n_ants)]
        self.antlions = [self.population_generator.new(self.problem) 
                        for _ in range(self.n_antlions)]
        return self.ants + self.antlions

    def evaluate(self, solution_list: List[S]) -> List[S]:
        return self.population_evaluator.evaluate(solution_list, self.problem)

    def init_progress(self) -> None:
        self.solutions = self.evaluate(self.solutions)
        self.ants = self.solutions[:self.n_ants]
        self.antlions = self.solutions[self.n_ants:]
        self.elite = min(self.antlions, key=lambda x: x.objectives[0])
        # print(f"Elite antlion: {self.elite.objectives[0]}")
        self.evaluations = self.n_ants

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def roulette_wheel_selection(self, fitness_values):
        min_fit = np.min(fitness_values)
        scaled_fitness = fitness_values - min_fit + 1e-6 
        probs = scaled_fitness / np.sum(scaled_fitness)
        return np.random.choice(range(len(fitness_values)), p=probs)

    def binary_random_walk(self, current_position: List[bool], mutation_rate: float) -> List[bool]:
        new_position = copy.deepcopy(current_position)
        for i in range(len(current_position)):
            if np.random.random() < mutation_rate:
                new_position[i] = not new_position[i]
        return new_position

    def binary_crossover(self, parent1: List[bool], parent2: List[bool]) -> List[bool]:
        child = []
        for i in range(len(parent1)):
            child.append(parent1[i] if np.random.random() < 0.5 else parent2[i])
        return child

    def calculate_mutation_rate(self):
        # print(self.evaluations, self.termination_criterion.max_evaluations)
        progress_ratio = self.evaluations / self.termination_criterion.max_evaluations
        return 0.1 + 0.9 * (1 - progress_ratio)
        # return 0.1 + 0.9 * np.exp(-2 * progress_ratio)

    def step(self):
        mutation_rate = self.calculate_mutation_rate()
        fitness_values = np.array([antlion.objectives[0] for antlion in self.antlions])
        
        # Para cada hormiga
        for i, ant in enumerate(self.ants):
            antlion_idx = self.roulette_wheel_selection(fitness_values)
            selected_antlion = self.antlions[antlion_idx]
            
            # Caminatas aleatorias binaria
            rw1 = self.binary_random_walk(selected_antlion.variables, mutation_rate)
            rw2 = self.binary_random_walk(self.elite.variables, mutation_rate)
            new_position = self.binary_crossover(rw1, rw2)
            new_ant = copy.deepcopy(ant)
            new_ant.variables = new_position
            new_ant = self.evaluate([new_ant])[0]
            
            # Actualizar hormiga si es mejor
            if new_ant.objectives[0] < ant.objectives[0]:
                self.ants[i] = new_ant
                
                # Actualizar antlion si es mejor
                if new_ant.objectives[0] < selected_antlion.objectives[0]:
                    self.antlions[antlion_idx] = copy.deepcopy(new_ant)
        
        new_elite = min(self.antlions, key=lambda x: x.objectives[0])
        if new_elite.objectives[0] < self.elite.objectives[0]:
            self.elite = copy.deepcopy(new_elite)
            # print(f"Updated elite fitness: {self.elite.objectives[0]}")
                    
                    # # Actualizar elite si es necesario
                    # if new_ant.objectives[0] < self.elite.objectives[0]:
                    #     self.elite = copy.deepcopy(new_ant)
                    #     print(f"Updated elite fitness: {self.elite.objectives[0]}")

    def update_progress(self) -> None:
        self.evaluations += self.n_ants
        # print(self.evaluations)
        observable_data = self.observable_data()
        self.observable.notify_all(**observable_data)

    def observable_data(self) -> dict:
        return {
            'PROBLEM': self.problem,
            'EVALUATIONS': self.evaluations,
            'SOLUTIONS': self.result(),
            'COMPUTING_TIME': time.time() - self.start_computing_time
        }

    def result(self) -> List[S]:
        # print(self.elite.objectives[0])
        return self.elite

    def get_name(self) -> str:
        return "Binary Ant Lion Optimizer"