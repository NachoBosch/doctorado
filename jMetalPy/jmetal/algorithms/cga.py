from typing import TypeVar, List

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
Module: Cellular genetic algorithm 
Synopsis: Implementation of Cellullar Genetic Algorithm mono-objective
"""
S = TypeVar("S")
R = TypeVar("R")

class CellularGeneticAlgorithm(EvolutionaryAlgorithm[S, R]):

    def __init__(self,
                 problem: Problem,
                 pop_size:int,
                 mutation: Mutation,
                 crossover: Crossover,
                 selection: Selection,
                 neighborhood: Neighborhood,
                 cell_update_policy: UpdatePolicy,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator):
        super(CellularGeneticAlgorithm, self).__init__(
            problem=problem,
            offspring_population_size=crossover.get_number_of_children(),
            population_size=pop_size
        )
        self.population_size=pop_size
        self.cell_update_policy = cell_update_policy
        self.neighborhood=neighborhood

        self.mutation_operator = mutation
        self.crossover_operator = crossover
        self.selection_operator = selection
        self.current_individual = 0
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.current_neighbors = []

        # self.best_individual_pos = 0

        self.comparator = DominanceComparator()

        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def create_initial_solutions(self) -> List[S]:
        # return [self.population_generator.new(self.problem) for _ in range(self.population_size)]
        return [self.problem.create_solution() for _ in range(self.population_size)]

    def evaluate(self, population: List[S]):
        # return self.population_evaluator.evaluate(population, self.problem)
        return [self.problem.evaluate(solution) for solution in population]
            

    def selection(self, population: List[S]) -> List[S]:
        parents = []
        self.current_neighbors = self.neighborhood.get_neighbors(self.current_individual, population)
        parents.append(self.selection_operator.execute(self.current_neighbors))
        parents.append(self.selection_operator.execute(self.current_neighbors))

        return parents

    def reproduction(self, mating_population: List[S]) -> List[S]:
        number_of_parents_to_combine = self.crossover_operator.get_number_of_parents()

        if len(mating_population) % number_of_parents_to_combine != 0:
            raise Exception('Wrong number of parents')

        offspring_population = self.crossover_operator.execute(mating_population)

        if offspring_population is None:
            offspring_population = mating_population.copy()

        for sol in offspring_population:
            self.mutation_operator.execute(sol)

        return offspring_population

    def replacement(self, population: [], offspring_population: []) -> []:
        offspring_population.sort(key=lambda s:s.objectives[0])
        if offspring_population[0].objectives[0] < population[self.current_individual].objectives[0]:
            population[self.current_individual] = offspring_population[0]
        return population

    def update_progress(self) -> None:
        self.evaluations += self.offspring_population_size
        print(self.evaluations)
        observable_data = self.observable_data()
        self.observable.notify_all(**observable_data)
        self.current_individual = (self.current_individual + 1) % self.population_size #Probar de implementar cell_update_policy

    def result(self) -> R:
        self.solutions.sort(key=lambda s: s.objectives[0])
        return self.solutions[0]

    def get_name(self) -> str:
        return 'Cellular Genetic Algorithm'