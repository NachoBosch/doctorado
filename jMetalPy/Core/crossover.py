from Solutions.solutions import BinarySolution
import random
import copy



class SPXCrossover():
    def __init__(self, probability: float):
        self.probability = probability

    def execute(self, parents: [BinarySolution]) -> [BinarySolution]:
        # print(f"Parents: {parents}\nLen: {len(parents)}")
        offspring = copy.deepcopy(parents)
        rand = random.random()

        if rand <= self.probability:
            # 1. Get the total number of bits
            total_number_of_bits = len(parents[0].variables)

            # 2. Calculate the point to make the crossover
            crossover_point = random.randrange(0, total_number_of_bits)

            # 3. Apply the crossover
            bitset1 = parents[0].variables[:]
            bitset2 = parents[1].variables[:]

            for i in range(crossover_point, total_number_of_bits):
                bitset1[i], bitset2[i] = bitset2[i], bitset1[i]

            offspring[0].variables = bitset1
            offspring[1].variables = bitset2

        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self) -> str:
        return "Single point crossover"
    
class TwoPointCrossover:
    def __init__(self, probability):
        self.probability = probability

    def execute(self, parents):
        if random.random() < self.probability:
            point1, point2 = sorted(random.sample(range(len(parents[0].variables)), 2))
            for i in range(point1, point2):
                parents[0].variables[i], parents[1].variables[i] = parents[1].variables[i], parents[0].variables[i]
        return parents
    
    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self) -> str:
        return "Single point crossover"