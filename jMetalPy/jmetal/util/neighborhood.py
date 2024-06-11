from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypeVar, Generic, List

import numpy

from jmetal.core.solution import Solution
from jmetal.util.ckecking import Check

"""
.. module:: neighborhood
   :platform: Unix, Windows
   :synopsis: implementation of neighborhoods in the context of list of solutions. The goal is,
   given the index of an element of the list, to find its neighbour solutions according to a criterion.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""

S = TypeVar('S')


class Neighborhood(Generic[S], ABC):

    @abstractmethod
    def get_neighbors(self, index: int, solution_list: List[S]) -> List[S]:
        pass


class TwoDimensionalMesh(Neighborhood):
    """
    Class defining a bi-mensional mesh.
    """

    def __init__(self, rows: int, columns: int, neighborhood: [[]]):
        self.rows = rows
        self.columns = columns
        self.neighborhood = neighborhood
        self.mesh = None
        self.__create_mesh()

    def __create_mesh(self):
        """ Example:
        if rows = 5, and columns=3, we need to fill the mesh as follows
        ----------
        |00-01-02|
        |03-04-05|
        |06-07-08|
        |09-10-11|
        |12-13-14|
        ----------
        """
        self.mesh = numpy.zeros((self.rows, self.columns), dtype=int)
        next_value = 0
        for i in range(self.rows):
            for j in range(self.columns):
                self.mesh[i][j] = next_value
                next_value += 1

    def __get_row(self, index: int) -> int:
        """
        Returns the row in the mesh where the index is local
        :param index:
        :return:
        """
        return index // self.columns

    def __get_column(self, index: int) -> int:
        """
        Returns the column in the mesh where the index is local
        :param index:
        :return:
        """
        return index % self.columns

    def __get_neighbor(self, index: int, neighbor: []) -> int:
        """
        Returns the neighbor of the index
        :param index:
        :param neighbor:
        :return:
        """

        row = self.__get_row(index)

        r = (row + neighbor[0]) % self.rows
        if r < 0:
            r = self.rows - 1

        column = self.__get_column(index)
        c = (column + neighbor[1]) % self.columns
        if c < 0:
            c = self.columns - 1

        return self.mesh[r][c]

    def __find_neighbors(self, solution_list: [], solution_index: int, neighborhood: [[]]):
        """
        Returns a list containing the neighbors of a given solution belongin to a solution list
        :param solution_list:
        :param solution_index:
        :param neighborhood:
        :return:
        """
        neighbors = []

        for neighbor in neighborhood:
            index = self.__get_neighbor(solution_index, neighbor=neighbor)
            neighbors.append(solution_list[index])

        return neighbors

    def get_neighbors(self, index: int, solution_list: List[Solution]) -> List[Solution]:
        Check.is_not_none(solution_list)
        Check.that(len(solution_list) != 0, "The list of solutions is empty")

        return self.__find_neighbors(solution_list, index, self.neighborhood)


class C9(TwoDimensionalMesh):
    """
    Class defining an C9 neighborhood of a solution belonging to a list of solutions which is
    structured as a bi-dimensional mesh. The neighbors are those solutions that are in 1-hop distance

   Shape:
           * * *
           * o *
           * * *

   Topology:
            north      = {-1,  0}
            south      = { 1 , 0}
            east       = { 0 , 1}
            west       = { 0 ,-1}
            north_east = {-1,  1}
            north_west = {-1, -1}
            south_east = { 1 , 1}
            south_west = { 1 ,-1}
    """

    def __init__(self, rows: int, columns: int):
        super(C9, self).__init__(rows, columns, [[-1, 0], [1, 0], [0, 1], [0, -1], [-1, 1], [-1, -1], [1, 1], [1, -1]])


class L5(TwoDimensionalMesh):
    """
    L5 neighborhood.
    Shape:
            *
          * o *
            *

    Topology:
        north = -1,  0
        south =  1,  0
        east  =  0,  1
        west  =  0, -1
    """

    def __init__(self, rows: int, columns: int):
        super(L5, self).__init__(rows, columns, [[-1, 0], [1, 0], [0, 1], [0, -1]])
