from typing import List, TypeVar
import numpy as np

from jmetal.core.operator import Selection
from jmetal.core.solution import Solution
from jmetal.util.comparator import Comparator, DominanceComparator
from jmetal.util.density_estimator import CrowdingDistance
from jmetal.util.ranking import FastNonDominatedRanking

S = TypeVar("S", bound=Solution)

"""
.. module:: selection
   :platform: Unix, Windows
   :synopsis: Module implementing selection operators.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class RouletteWheelSelection(Selection[List[S], S]):
    """Performs roulette wheel selection."""

    def __init__(self):
        super(RouletteWheelSelection).__init__()

    def execute(self, front: List[S]) -> S:
        if front is None:
            raise Exception("The front is null")
        elif len(front) == 0:
            raise Exception("The front is empty")

        maximum = sum([solution.objectives[0] for solution in front])
        rand = np.random.uniform(0.0, maximum)
        value = 0.0

        for solution in front:
            value += solution.objectives[0]

            if value > rand:
                return solution

        return None

    def get_name(self) -> str:
        return "Roulette wheel selection"


class BinaryTournamentSelection(Selection[List[S], S]):
    def __init__(self, comparator: Comparator = DominanceComparator()):
        super(BinaryTournamentSelection, self).__init__()
        self.comparator = comparator

    def execute(self, front: List[S]) -> S:
        if front is None:
            raise Exception("The front is null")
        elif len(front) == 0:
            raise Exception("The front is empty")

        if len(front) == 1:
            result = front[0]
        else:
            # Sampling without replacement
            i, j = np.random.choice(len(front), 2, replace=False)
            solution1 = front[i]
            solution2 = front[j]

            flag = self.comparator.compare(solution1, solution2)

            if flag == -1:
                result = solution1
            elif flag == 1:
                result = solution2
            else:
                result = [solution1, solution2][np.random.random() < 0.5]

        return result

    def get_name(self) -> str:
        return "Binary tournament selection"

class DifferentialEvolutionSelection(Selection[List[S], List[S]]):
    def __init__(self):
        super(DifferentialEvolutionSelection, self).__init__()
        self.index_to_exclude = None

    def execute(self, front: List[S]) -> List[S]:
        if front is None:
            raise Exception("The front is null")
        elif len(front) == 0:
            raise Exception("The front is empty")
        elif len(front) < 4:
            raise Exception("The front has less than four solutions: " + str(len(front)))

        selected_indexes = np.random.choice(range(len(front)), 3, replace=False)
        while self.index_to_exclude in selected_indexes:
            selected_indexes = np.random.choice(range(len(front)), 3, replace=False)

        return [front[i] for i in selected_indexes]

    def set_index_to_exclude(self, index: int):
        self.index_to_exclude = index

    def get_name(self) -> str:
        return "Differential evolution selection"