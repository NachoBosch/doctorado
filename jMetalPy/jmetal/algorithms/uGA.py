import copy
from typing import TypeVar, List

import numpy as np
import random
import scipy.stats as stats

from jmetal.config import store
from jmetal.core.algorithm import EvolutionaryAlgorithm
from jmetal.core.problem import Problem
from jmetal.core.operator import Mutation, Crossover, Selection
from jmetal.util.comparator import DominanceComparator
from jmetal.util.evaluator import Evaluator
from jmetal.util.generator import Generator
from jmetal.util.termination_criterion import TerminationCriterion
from jmetal.util.update_policy import UpdatePolicy
from jmetal.util.neighborhood import Neighborhood

"""
Module: Micro genetic algorithm 
Synopsis: Implementation of Micro Genetic Algorithm mono-objective
"""
S = TypeVar('S')
R = TypeVar('R')

class MicroGeneticAlgorithm(EvolutionaryAlgorithm[S, R]):

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 mutation: Mutation,
                 crossover: Crossover,
                 selection: Selection,
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

    def selection(self, population: []) -> []:
        parents = []
        parents.append(self.selection_operator.execute(population))
        parents.append(self.selection_operator.execute(population))
        return parents

    def reproduction(self, mating_population: []) -> []:
        number_of_parents_to_combine = self.crossover_operator.get_number_of_parents()
        if len(mating_population) % number_of_parents_to_combine != 0:
            raise Exception('Wrong number of parents')
        offspring_population = self.crossover_operator.execute(mating_population)
        if offspring_population is None:
            offspring_population = copy.deepcopy(mating_population)
        for sol in offspring_population:
            self.mutation_operator.execute(sol)
        return offspring_population

    def replacement(self, population: [], offspring_population: []) -> []:
        population.extend(offspring_population)
        population.sort(key = lambda s:s.objectives[0])
        population = population[: self.population_size]
        if self.evaluations % self.reinicio == 0:
            # print("Re-initialization population")
            self.reinition_population(population)
        return population
    
    def reinition_population(self, population):
        best_solution = min(population, key=lambda s:s.objectives[0])
        for i in range(len(population)):
            if population[i] is not best_solution:
                population[i] = self.problem.create_solution()
                self.problem.evaluate(population[i])

    def update_progress(self):
        self.evaluations += self.offspring_population_size
        observable_data = self.observable_data()
        self.observable.notify_all(**observable_data)

    def result(self):
        return self.solutions[0]

    def get_name(self) -> str:
        return 'Micro Genetic Algorithm'