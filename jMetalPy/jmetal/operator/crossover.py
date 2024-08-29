# import copy
import numpy as np
from typing import List

from jmetal.core.operator import Crossover
from jmetal.core.solution import (
    BinarySolution,
    CompositeSolution,
    FloatSolution,
    IntegerSolution,
    PermutationSolution,
    Solution,
)
from jmetal.util.ckecking import Check

"""
.. module:: crossover
   :platform: Unix, Windows
   :synopsis: Module implementing crossover operators.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class NullCrossover(Crossover[Solution, Solution]):
    def __init__(self):
        super(NullCrossover, self).__init__(probability=0.0)

    def execute(self, parents: List[Solution]) -> List[Solution]:
        if len(parents) != 2:
            raise Exception("The number of parents is not two: {}".format(len(parents)))

        return parents

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self):
        return "Null crossover"

class SPXCrossover(Crossover[BinarySolution, BinarySolution]):
    def __init__(self, probability: float):
        super(SPXCrossover, self).__init__(probability=probability)

    def execute(self, parents: List[BinarySolution]) -> List[BinarySolution]:
        Check.that(type(parents[0]) is BinarySolution, "Solution type invalid")
        Check.that(type(parents[1]) is BinarySolution, "Solution type invalid")
        Check.that(len(parents) == 2, "The number of parents is not two: {}".format(len(parents)))

        offspring = parents.copy()
        rand = np.random.random()

        if rand <= self.probability:
            # 1. Get the total number of bits
            total_number_of_bits = parents[0].get_total_number_of_bits()

            # 2. Calculate the point to make the crossover
            crossover_point = np.random.randint(0, total_number_of_bits)

            # 3. Compute the variable containing the crossover bit
            variable_to_cut = 0
            bits_count = len(parents[1].variables[variable_to_cut])
            while bits_count < (crossover_point + 1):
                variable_to_cut += 1
                bits_count += len(parents[1].variables[variable_to_cut])

            # 4. Compute the bit into the selected variable
            diff = bits_count - crossover_point
            crossover_point_in_variable = len(parents[1].variables[variable_to_cut]) - diff

            # 5. Apply the crossover to the variable
            bitset1 = parents[0].variables[variable_to_cut].copy()
            bitset2 = parents[1].variables[variable_to_cut].copy()

            for i in range(crossover_point_in_variable, len(bitset1)):
                swap = bitset1[i]
                bitset1[i] = bitset2[i]
                bitset2[i] = swap

            offspring[0].variables[variable_to_cut] = bitset1
            offspring[1].variables[variable_to_cut] = bitset2

            # 6. Apply the crossover to the other variables
            for i in range(variable_to_cut + 1, len(parents[0].variables)):
                offspring[0].variables[i] = parents[1].variables[i].copy()
                offspring[1].variables[i] = parents[0].variables[i].copy()

        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self) -> str:
        return "Single point crossover"
    
class DifferentialEvolutionCrossover(Crossover[BinarySolution, BinarySolution]):
    """This operator receives two parameters: the current individual and an array of three parent individuals. The
    best and rand variants depends on the third parent, according whether it represents the current of the "best"
    individual or a random_search one. The implementation of both variants are the same, due to that the parent selection is
    external to the crossover operator.
    """

    def __init__(self, CR: float, F: float, K: float = 0.5):
        super(DifferentialEvolutionCrossover, self).__init__(probability=1.0)
        self.CR = CR
        self.F = F
        self.K = K

        self.current_individual: BinarySolution = None

    def execute(self, parents: List[BinarySolution]) -> List[BinarySolution]:
        """Execute the differential evolution crossover ('best/1/bin' variant in jMetal)."""
        if len(parents) != self.get_number_of_parents():
            raise Exception("The number of parents is not {}: {}".format(self.get_number_of_parents(), len(parents)))

        child = self.current_individual# volver a usar el deepcopy 

        number_of_variables = len(parents[0].variables)
        rand = np.random.randint(0, number_of_variables - 1)

        for i in range(number_of_variables):
            if np.random.random() < self.CR or i == rand:
                value = parents[2].variables[i] + self.F * (parents[0].variables[i] - parents[1].variables[i])

                if value < child.lower_bound[i]:
                    value = child.lower_bound[i]
                if value > child.upper_bound[i]:
                    value = child.upper_bound[i]
            else:
                value = child.variables[i]

            child.variables[i] = value

        return [child]

    def get_number_of_parents(self) -> int:
        return 3

    def get_number_of_children(self) -> int:
        return 1

    def get_name(self) -> str:
        return "Differential Evolution crossover"

class DifferentialBinaryEvolutionCrossover(Crossover[BinarySolution, BinarySolution]):
    """This operator receives two parameters: the current individual and an array of three parent individuals. The
    best and rand variants depends on the third parent, according whether it represents the current of the "best"
    individual or a random_search one. The implementation of both variants are the same, due to that the parent selection is
    external to the crossover operator.
    """

    def __init__(self, CR: float, F: float):
        super(DifferentialBinaryEvolutionCrossover, self).__init__(probability=1.0)
        self.CR = CR
        self.F = F
        self.current_individual:BinarySolution = None

    def execute(self, parents: List[BinarySolution]) -> List[BinarySolution]:
        if len(parents) != self.get_number_of_parents():
            raise Exception(f"The number of parents is not {self.get_number_of_parents()}: {len(parents)}")

        p1, p2, p3 = parents[0], parents[1], parents[2]
        number_of_variables = len(p1.variables)

        target = self.current_individual.copy()

        for i in range(number_of_variables):
            if np.random.rand() < self.CR:
                if p2.variables[i] != p3.variables[i]:
                    if np.random.rand() < self.F:
                        target.variables[i] = not p1.variables[i]
                    else:
                        target.variables[i] = p1.variables[i]
                else:
                    target.variables[i] = p2.variables[i]

        return [target]

    def get_number_of_parents(self) -> int:
        return 3

    def get_number_of_children(self) -> int:
        return 1

    def get_name(self) -> str:
        return "Differential Binary Evolution crossover"