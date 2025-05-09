from jmetal.core.algorithm import EvolutionaryAlgorithm
from jmetal.core.problem import Problem
from jmetal.core.operator import Mutation, Crossover, Selection
from jmetal.util.comparator import DominanceComparator
from jmetal.util.generator import Generator
from jmetal.util.evaluator import Evaluator
from jmetal.config import store
from jmetal.util.termination_criterion import TerminationCriterion
from typing import TypeVar, List

S = TypeVar('S')
R = TypeVar('R')

class GeneticAlgorithm(EvolutionaryAlgorithm[S, R]):

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 offspring_population_size: int,
                 mutation: Mutation,
                 crossover: Crossover,
                 selection: Selection,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator):
        
        super(GeneticAlgorithm, self).__init__(
            problem=problem,
            population_size=population_size,
            offspring_population_size=offspring_population_size
        )

        self.mutation_operator = mutation
        self.crossover_operator = crossover
        self.selection_operator = selection
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)

        self.comparator = DominanceComparator()

        self.mating_pool_size = (
            self.offspring_population_size
            * self.crossover_operator.get_number_of_parents()
            // self.crossover_operator.get_number_of_children()
        )

        if self.mating_pool_size < self.crossover_operator.get_number_of_children():
            self.mating_pool_size = self.crossover_operator.get_number_of_children()

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met
    
    def create_initial_solutions(self) -> []:
        return [self.population_generator.new(self.problem) for _ in range(self.population_size)]

    def evaluate(self, population: []) -> []:
        return self.population_evaluator.evaluate(population, self.problem)

    def selection(self, population: []) -> []:
        return [self.selection_operator.execute(population) for _ in range(self.mating_pool_size)]

    def reproduction(self, mating_population: List[S]) -> List[S]:
        number_of_parents_to_combine = self.crossover_operator.get_number_of_parents()
        # print(f"Number of parents to combine: {number_of_parents_to_combine}")

        if len(mating_population) % number_of_parents_to_combine != 0:
            raise Exception("Wrong number of parents for crossover")

        offspring_population = []
        for i in range(0, self.mating_pool_size, number_of_parents_to_combine):
            parents = mating_population[i:i + number_of_parents_to_combine]
            offspring = self.crossover_operator.execute(parents)

            for solution in offspring:
                self.mutation_operator.execute(solution)
                offspring_population.append(solution)

                if len(offspring_population) >= self.offspring_population_size:
                    break

        return offspring_population

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[S]:
        combined = population + offspring_population
        combined.sort(key=lambda s: s.objectives[0])  # Minimization Problem
        return combined[:self.population_size]

    def update_progress(self):
        self.evaluations += self.offspring_population_size
        print(self.evaluations)
        observable_data = self.observable_data()
        self.observable.notify_all(**observable_data)

    def result(self) -> S:
        return self.solutions[0]

    def get_name(self) -> str:
        return "Simple Genetic Algorithm"
