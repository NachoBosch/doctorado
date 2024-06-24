import numpy as np
from jmetal.core.solution import BinarySolution

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
            i, j = np.random.choice(len(front), 2, replace=False)
            solution1 = front[i]
            solution2 = front[j]

            flag = self.compare(solution1, solution2)

            if flag == -1:
                result = solution1
            elif flag == 1:
                result = solution2
            else:
                result = [solution1, solution2][np.random.random() < 0.5]
        return result
    
    def compare(self, solution1: BinarySolution, solution2: BinarySolution) -> int:
        return self.dominance_test(solution1.objectives[0], solution2.objectives[0])

    def dominance_test(self,vector1, vector2) -> int:
        result = 0
        if vector1 != vector2:
            if vector1 < vector2:
                result = -1
            if vector1 > vector2:
                result = 1
        return result

    def get_name(self) -> str:
        return "Binary tournament selection"