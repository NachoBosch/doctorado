from Solutions.solutions import BinarySolution

class DominanceComparator():
    def compare(self, solution1: BinarySolution, solution2: BinarySolution) -> int:
        return self.dominance_test(solution1.objectives, solution2.objectives[0])

    def dominance_test(self,vector1, vector2) -> int:
        result = 0
        for i in range(len(vector1)):
            if vector1[i] > vector2[i]:
                if result == -1:
                    return 0
                result = 1
            elif vector2[i] > vector1[i]:
                if result == 1:
                    return 0
                result = -1

        return result