from Solutions.solutions import BinarySolution
import random
import copy



class SPXCrossover():
    def __init__(self, probability: float):
        self.probability = probability

    def execute(self, parents: [BinarySolution]) -> [BinarySolution]:
        # print(f"Parents: {parents}\nLen: {len(parents)}")
        rand = random.random()
        parent1, parent2 = parents
        offspring1 = parent1
        offspring2 = parent2

        if rand <= self.probability:
            crossover_point = random.randint(0, len(parent1.variables))
            offspring1.variables[:crossover_point] = parent1.variables[:crossover_point]
            offspring1.variables[crossover_point:] = parent2.variables[crossover_point:]
            offspring2.variables[:crossover_point] = parent2.variables[:crossover_point]
            offspring2.variables[crossover_point:] = parent1.variables[crossover_point:]

        return [offspring1, offspring2]

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self) -> str:
        return "Single point crossover"