from Solutions.solutions import BinarySolution
import random


class BitFlipMutation():

    def __init__(self, probability: float):
        
        self.probability = probability

    def execute(self, solution: BinarySolution) -> BinarySolution:

        for i in range(solution.number_of_variables):
            rand = random.random()
            if rand <= self.probability:
                solution.variables[i] = 1 if solution.variables[i] is 0 else 0
        return solution

    def get_name(self):
        return "BitFlip mutation"