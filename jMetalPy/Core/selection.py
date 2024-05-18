import random
from Solutions.solutions import BinarySolution

class BinaryTournamentSelection():
    def __init__(self):
        pass

    def execute(self, front: []) -> []:
        if front is None:
            raise Exception("The front is null")
        elif len(front) == 0:
            raise Exception("The front is empty")

        if len(front) == 1:
            result = front[0]
        else:
            # Sampling without replacement
            i, j = random.sample(range(0, len(front)), 2)
            solution1 = front[i]
            solution2 = front[j]

            flag = self.compare(solution1, solution2)

            if flag == -1:
                result = solution1
            elif flag == 1:
                result = solution2
            else:
                result = [solution1, solution2][random.random() < 0.5]
        return result
    
    def compare(self, solution1: BinarySolution, solution2: BinarySolution) -> int:
        return self.dominance_test(solution1.objectives[0], solution2.objectives[0])

    def dominance_test(self,vector1, vector2) -> int:
        result = 0
        # for i in range(len(vector1)):
            # print("Vector1",vector1)
            # print("Vector2",vector2)
        if vector1 != vector2:
            if vector1 < vector2:
                result = -1
            if vector1 > vector2:
                result = 1
        # print("Compare solutions: ",result)
        return result

    def get_name(self) -> str:
        return "Binary tournament selection"