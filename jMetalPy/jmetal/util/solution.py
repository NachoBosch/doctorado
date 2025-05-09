import logging
import os
from pathlib import Path
from typing import List

from jmetal.core.solution import FloatSolution, Solution, BinarySolution
from jmetal.util.archive import Archive, NonDominatedSolutionsArchive

logger = logging.getLogger(__name__)


"""
.. module:: solutions
   :platform: Unix, Windows
   :synopsis: Utils to print solutions.

.. moduleauthor:: Antonio J. Nebro <ajnebro@uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


def get_non_dominated_solutions(solutions: List[Solution]) -> List[Solution]:
    archive: Archive = NonDominatedSolutionsArchive()

    for solution in solutions:
        archive.add(solution)

    return archive.solution_list


def read_solutions(filename: str) -> List[FloatSolution]:
    """Reads a reference front from a file.

    :param filename: File path where the front is located.
    """
    front = []

    if Path(filename).is_file():
        with open(filename) as file:
            for line in file:
                # print(f"\n FITNESS --->: {line}\n")
                vector = [float(line.split()[0])]

                solution = FloatSolution([], [], len(vector))
                solution.objectives = vector

                front.append(solution)
    else:
        logger.warning("Reference front file was not found at {}".format(filename))

    return front

def read_binary_solutions(filename: str) -> List[BinarySolution]:
    """Reads binary solutions from a file.

    :param filename: File path where the solutions are located.
    """
    solutions = []

    if Path(filename).is_file():
        with open(filename) as file:
            for line in file:
                vector = [eval(x) for x in line.split()]
                solution = BinarySolution(len(vector), 0)
                solution.variables = vector

                solutions.append(solution)
    else:
        logger.warning("Solution file was not found at {}".format(filename))

    return solutions

def read_accuracy_values(filename: str) -> List[Solution]:
    solutions = []
    with open(filename, "r") as file:
        for line in file:
            # print(f"\n {line.split()[1]}")
            vector = [float(line.split()[1])]
            solution = FloatSolution([], [], len(vector))
            solution.objectives = vector
            solutions.append(solution)
    return solutions

def print_variables_to_file(solutions, filename: str):
    logger.info("Output file (variables): " + filename)

    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    except FileNotFoundError:
        pass

    if type(solutions) is not list:
        solutions = [solutions]

    with open(filename, "w") as of:
        for solution in solutions:
            # print(f"Print variables: {len(solution.variables)}")
            for variables in solution.variables:
                of.write(str(variables) + " ")
            of.write("\n")


def print_variables_to_screen(solutions):
    if type(solutions) is not list:
        solutions = [solutions]

    for solution in solutions:
        print(solution.variables[0])


def print_function_values_to_file(solutions, filename: str):
    logger.info("Output file (function values): " + filename)

    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    except FileNotFoundError:
        pass

    if type(solutions) is not list:
        solutions = [solutions]

    with open(filename, "w") as of:
        for solution in solutions:
            for function_value in solution.objectives:
                of.write(str(function_value) + " ")
            of.write("\n")


def print_function_values_to_screen(solutions):
    if type(solutions) is not list:
        solutions = [solutions]

    for solution in solutions:
        print(str(solutions.index(solution)) + ": ", sep="  ", end="", flush=True)
        print(solution.objectives, sep="  ", end="", flush=True)
        print()
