from typing import List, TypeVar
from jmetal.config import store
from jmetal.core.algorithm import  EvolutionaryAlgorithm
from jmetal.core.problem import Problem
from jmetal.core.solution import FloatSolution
from jmetal.operator.crossover import DifferentialBinaryEvolutionCrossover
from jmetal.operator.selection import DifferentialEvolutionSelection
from jmetal.util.comparator import Comparator
from jmetal.util.evaluator import Evaluator
from jmetal.util.generator import Generator
from jmetal.util.termination_criterion import TerminationCriterion

S = TypeVar("S")
R = List[S]


class DE(EvolutionaryAlgorithm[S,R]):
    def __init__(
        self,
        problem: Problem,
        population_size: int,
        F: float,
        CR: float,
        termination_criterion: TerminationCriterion = store.default_termination_criteria,
        population_generator: Generator = store.default_generator,
        population_evaluator: Evaluator = store.default_evaluator,
        dominance_comparator: Comparator = store.default_comparator,
    ):
        super(DE, self).__init__(problem=problem, population_size=population_size, offspring_population_size=population_size)
        self.dominance_comparator = dominance_comparator
        self.selection_operator = DifferentialEvolutionSelection()
        self.crossover_operator = DifferentialBinaryEvolutionCrossover(F, CR)

        self.population_generator = population_generator
        self.population_evaluator = population_evaluator

        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)

    def selection(self, population: []) -> []:
        mating_pool = []
        for i in range(self.population_size):
            self.selection_operator.set_index_to_exclude(i)
            selected_solutions = self.selection_operator.execute(self.solutions)
            mating_pool = mating_pool + selected_solutions
        return mating_pool

    def reproduction(self, mating_pool: []) -> []:
        offspring_population = []
        # first_parent_index = 0
        print(f"Mating pool: {len(mating_pool)}")
        print(f"Sel solution: {len(self.solutions)}")
        for i in range(self.population_size): 
            self.crossover_operator.current_individual = self.solutions[i]
            parents = mating_pool[i*3:(i+1)*3]
            # print(solution)
            # self.crossover_operator.current_individual = solution
            # parents = mating_pool[first_parent_index:first_parent_index+3]
            # print(f"Parents: {parents}")
            # parents = np.random.choice(parents,3,replace=False)
            # print(f"Parents: {len(parents)}")
            offspring = self.crossover_operator.execute(parents)[0]
            offspring_population.append(offspring)
        # print(len(offspring_population))
        return offspring_population

    def replacement(self, population: [], offspring_population: []) -> [[]]:
        new_population = []

        for solution1, solution2 in zip(self.solutions, offspring_population):
            result = self.dominance_comparator.compare(solution1, solution2)
            if result == -1:
                new_population.append(solution1)
            elif result == 1:
                new_population.append(solution2)
            else:
                new_population.append(solution1)
                new_population.append(solution2)
        
        return new_population[:self.population_size]

    def create_initial_solutions(self) -> []:
        return [self.population_generator.new(self.problem) for _ in range(self.population_size)]

    def evaluate(self, solution_list: []) -> []:
        return self.population_evaluator.evaluate(solution_list, self.problem)

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def result(self) -> []:
        return self.solutions[0]

    def get_name(self) -> str:
        return "DE"