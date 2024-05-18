


class BinarySolution():
    """Class representing float solutions"""

    def __init__(self, number_of_variables: int, number_of_objectives: int, number_of_constraints: int = 0):
        self.number_of_variables = number_of_variables
        self.number_of_objectives = number_of_objectives
        self.number_of_constraints = number_of_constraints
        self.objectives = []
        
    def __copy__(self):
        new_solution = BinarySolution(self.number_of_variables, self.number_of_objectives)
        new_solution.objectives = self.objectives[:]
        new_solution.variables = self.variables[:]

        new_solution.attributes = self.attributes.copy()

        return new_solution

    def get_total_number_of_bits(self) -> int:
        total = 0
        print("Variable",type(self.variables))
        #print(self.variables)
        # total = len(self.variables)
        for var in self.variables:
            total += len(var)
        return total

    def get_binary_string(self) -> str:
        string = ""
        for bit in self.variables[0]:
            string += "1" if bit else "0"
        return string
