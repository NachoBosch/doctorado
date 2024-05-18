

class CellularGeneticAlgorithm():
    def __init__(
            self,
            problem,
            population_size: int,
            offspring_population_size: int,
            max_evaluations:int,
            mutation,
            crossover,
            selection):
        
        self.problem = problem
        self.population_size = population_size
        self.offspring_population_size = offspring_population_size
        self.mutation_operator = mutation
        self.crossover_operator = crossover
        self.selection_operator = selection
        self.max_evaluations = max_evaluations
        self.evaluations = 0

        self.cell_structure = None
        self.mating_pool_size = (
            self.offspring_population_size
            * self.crossover_operator.get_number_of_parents()
            // self.crossover_operator.get_number_of_children()
        )

        if self.mating_pool_size < self.crossover_operator.get_number_of_children():
            self.mating_pool_size = self.crossover_operator.get_number_of_children()


    def run(self):
        print("EVOLVE")
        self.population = self.create_initial_solutions()
        self.evaluate(self.population)
        self.initialize_cell_structure()
        i = 0
        print(f"Progress: {self.evaluations}/{self.max_evaluations}, Fitness: {self.get_result().objectives[0]}\n")
        while not self.stopping_condition_is_met():
            print(f"Progress: {self.evaluations}/{self.max_evaluations}, Fitness: {self.get_result().objectives[0]}\n")   
            offspring_population = self.reproduction(self.selection(self.population))
            self.evaluate(offspring_population)
            self.update_cell_structure(offspring_population)
            self.population = self.replacement(self.population, offspring_population)
            i+=1

    def create_initial_solutions(self) -> []:
        print("Create initial solution")
        print("Pop size:", self.population_size)
        return [self.problem.create_solution() for _ in range(self.population_size)]

    def initialize_cell_structure(self):
        self.cell_structure = [0] * self.population_size

    def evaluate(self, population):
        print("Evaluate algorithm population",len(population))
        # evaluate = []
        for solution in population:
            # evaluate.append(self.problem.evaluate(solution))
            self.problem.evaluate(solution)
        self.evaluations+=1
        #print(len(evaluate))
        #return evaluate

    def stopping_condition_is_met(self) -> bool:
        return self.evaluations >= self.max_evaluations

    def selection(self, population:[]):
        mating_population = []

        for _ in range(self.mating_pool_size):
            solution = self.selection_operator.execute(population)
            mating_population.append(solution)
        print("Mating population for reproduction:",len(mating_population))
        return mating_population

    def reproduction(self, mating_population: []) -> []:
        number_of_parents_to_combine = self.crossover_operator.get_number_of_parents()

        if len(mating_population) < number_of_parents_to_combine:
            raise ValueError("Insufficient individuals in mating population for reproduction")

        offspring_population = []
        for i in range(0, len(mating_population), number_of_parents_to_combine):
            parents = []
            for j in range(number_of_parents_to_combine):
                index = i + j

                if index < len(mating_population):
                    parents.append(mating_population[index])
            if len(parents) == number_of_parents_to_combine:
                offspring = self.crossover_operator.execute(parents)
                for solution in offspring:
                    self.mutation_operator.execute(solution)
                    offspring_population.append(solution)
                    if len(offspring_population) >= self.offspring_population_size:
                        break
        return offspring_population

    def update_cell_structure(self, population: []):
        population_size = len(population)
        for i, solution in enumerate(population):
            neighbors = [population[j] for j in range(population_size) if j != i]
            avg_fitness = sum(neighbor.objectives[0] for neighbor in neighbors) / len(neighbors)
            self.cell_structure[i] = avg_fitness
    
    def replacement(self, population, offspring_population):
        combined_population = population + offspring_population
        combined_population.sort(key=lambda sol: sol.objectives[0])
        return combined_population[:self.population_size]

    def get_result(self):
        self.population.sort(key=lambda s: s.objectives[0])
        return self.population[0]

    def get_name(self) -> str:
        return "Cellular Genetic Algorithm"