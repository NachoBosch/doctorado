import copy
from typing import TypeVar

from config import store
from core.algorithm import ParticleSwarmOptimization
from core.problem import Problem
from core.operator import Mutation, Crossover, Selection
from util.comparator import DominanceComparator
from util.evaluator import Evaluator
from util.generator import Generator
from util.termination_criterion import TerminationCriterion

"""
Module: Cellular genetic algorithm 
Synopsis: Implementation of Cellullar Genetic Algorithm mono-objective
"""
S = TypeVar('S')
R = TypeVar('R')

class MicroGeneticAlgorithm(ParticleSwarmOptimization[S, R]):

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 inertia:int,
                 
                 reinicio: int,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator):
        
        super(MicroGeneticAlgorithm, self).__init__(
            problem=problem,
            offspring_population_size=crossover.get_number_of_children(),
            population_size=population_size
        )
        self.reinicio = reinicio

        self.mutation_operator = mutation
        self.crossover_operator = crossover
        self.selection_operator = selection
        self.current_individual = 0
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.current_neighbors = []

        self.best_individual_pos = 0

        self.comparator = DominanceComparator()

        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def create_initial_solutions(self) -> []:
        return [self.problem.create_solution()
                for _ in range(self.population_size)]

    def evaluate(self, population: []):
        return self.population_evaluator.evaluate(population, self.problem)

    def initialize_velocity(self):
        pass
    
    def initialize_particle_best(self, swarm: []):
        pass

    def initialize_global_best(self, swarm: []):
        pass

    def update_velocity(self, swarm: []):
        pass

    def update_particle_best(self, swarm: []):
        pass

    def update_global_best(self, swarm: []):
        pass

    def update_position(self, swarm: []):
        pass

    def perturbation(self, swarm: []):
        pass

    def update_progress(self):
        self.evaluations += self.offspring_population_size
        observable_data = self.observable_data()
        self.observable.notify_all(**observable_data)

    def result(self):
        return self.solutions[0]

    def get_name(self) -> str:
        return 'Particle Swarm Algorithm'