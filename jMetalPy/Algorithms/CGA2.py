import random
import matplotlib.pyplot as plt

class CellularGeneticAlgorithm():
    def __init__(
            self,
            problem,
            population_size: int,
            offspring_population_size: int,
            max_evaluations: int,
            mutation,
            crossover,
            selection,
            neighborhood_size: int = 10):
        
        self.problem = problem
        self.population_size = population_size
        self.offspring_population_size = offspring_population_size
        self.mutation_operator = mutation
        self.crossover_operator = crossover
        self.selection_operator = selection
        self.max_evaluations = max_evaluations
        self.evaluations = 0
        self.neighborhood_size = neighborhood_size
        self.best_fitness_per_epoch = [] 

    def run(self):
        print("EVOLVE")
        self.population = self.create_initial_solutions()
        self.fitness_values = self.evaluate(self.population)
        self.best_fitness_per_epoch.append(self.get_result().objectives[0])
        # print(f"Progress: {self.evaluations}/{self.max_evaluations}, Fitness: {self.get_result().objectives[0]}\n")
        while not self.stopping_condition_is_met():
            print(f"Progress: {self.evaluations}/{self.max_evaluations}, Fitness: {self.get_result().objectives[0]}\n")   
            mates = self.select_parents(self.population,self.fitness_values)
            offspring_population = self.reproduction(mates)
            self.fitness_values = self.evaluate(offspring_population)
            self.population = self.replacement(self.population, offspring_population)
            self.best_fitness_per_epoch.append(self.get_result().objectives[0])

    def create_initial_solutions(self) -> []:
        print("Create initial solution")
        print("Pop size:", self.population_size)
        return [self.problem.create_solution() for _ in range(self.population_size)]

    def evaluate(self, population):
        print("Evaluate algorithm population", len(population))
        fitness_values = []
        for solution in population:
            eval_solution = self.problem.evaluate(solution)
            fitness_values.append(eval_solution)
        self.evaluations += 1
        return fitness_values

    def stopping_condition_is_met(self) -> bool:
        return self.evaluations >= self.max_evaluations

    def select_neighborhood(self, index):
        neighbors = []
        i = index
        # for i in range(len(population)):
        x,y = i//self.neighborhood_size, i%self.neighborhood_size
        neighbor_index = [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]
        for nx, ny in neighbor_index:
            if 0<=nx<self.neighborhood_size and 0<=ny<self.neighborhood_size:
                neighbors.append(nx*self.neighborhood_size+ny)
        return neighbors

    def tournament_selection(self, neighborhood,fitness_values):
        best_index = neighborhood[0]
        best_fitness = fitness_values[best_index]
        for i in neighborhood:
            fitness = fitness_values[i]
            if fitness.objectives[0] < best_fitness.objectives[0]:
                best_index = i
        return best_index

    def select_parents(self, population,fitness_values):
        select_parents=[]
        for i in range(len(population)):
            neigborhood = self.select_neighborhood(i)
            parent1_index =  self.tournament_selection(neigborhood,fitness_values)
            parent2_index =  self.tournament_selection(neigborhood,fitness_values)
            select_parents.append(population[parent1_index])
            select_parents.append(population[parent2_index])
        # print(select_parents)
        return select_parents

    def reproduction(self, mates):
        number_of_parents_to_combine = self.crossover_operator.get_number_of_parents()

        if len(mates) < number_of_parents_to_combine:
            raise ValueError("Insufficient individuals in mating population for reproduction")

        offspring_population = []
        for i in range(0, len(mates), number_of_parents_to_combine):
            parents = []
            for j in range(number_of_parents_to_combine):
                index = i + j
                if index < len(mates):
                    parents.append(mates[index])
            if len(parents) == number_of_parents_to_combine:
                offspring = self.crossover_operator.execute(parents)
                for solution in offspring:
                    self.mutation_operator.execute(solution)
                    offspring_population.append(solution)
                    if len(offspring_population) >= self.offspring_population_size:
                        break
        return offspring_population

    def replacement(self,population,offspring_population):
        combined_population = population + offspring_population
        combined_population.sort(key=lambda sol: sol.objectives[0])
        return combined_population[:self.population_size]
    
    def get_result(self):
        self.population.sort(key=lambda s: s.objectives[0])
        return self.population[0]
    
    def plot_fitness(self):
        plt.figure()
        plt.plot(self.best_fitness_per_epoch)
        plt.xlabel("Epoch")
        plt.ylabel("Fitness")
        plt.title("Fitness progress")
        plt.show()


    def get_name(self) -> str:
        return "Cellular Genetic Algorithm"