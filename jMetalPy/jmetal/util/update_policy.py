from abc import ABC, abstractmethod
from typing import List


class UpdatePolicy(ABC):

    @abstractmethod
    def get_order(self, pop_grid: List) -> List:
        """
        :param pop_grid: population matriz that is used to define
        the update order.
        :return: An ordered list of tuples, that contains the position of the
        individuals to be updated, ordered by random form.

        Update policy, it's based on random order of update and
        it's allows replacement.
        """
        pass


class LineSweep(UpdatePolicy):
    def __init__(self):
        pass

    def get_order(self, pop_grid: List) -> List:
        return [x for x in range(len(pop_grid))]


class RandomSweep(UpdatePolicy):
    def __init__(self):
        pass

    def get_order(self, pop_grid: List) -> List:
        pass


class UniformChoice(UpdatePolicy):
    def __init__(self):
        pass

    def get_order(self, pop_grid: List) -> List:
        pass