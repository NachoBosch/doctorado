import random
import matplotlib.pyplot as plt
import pandas as pd

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
            neighborhood_size:int):
        
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
        self.min_variables_per_epoch = []
        self.epochs = self.max_evaluations//self.population_size 

    def run(self):
        self.population = self.create_initial_solutions()
        self.fitness_values = self.evaluate(self.population)
        self.best_fitness_per_epoch.append(self.get_result().objectives[0])
        self.min_variables_per_epoch.append(sum(self.get_result().variables))
        print(f"Epochs: {self.evaluations}/{self.epochs}, Fitness: {self.get_result().objectives[0]}\n")

        while not self.stopping_condition_is_met():

            for i in range(len(self.population)):
                neighborhood = self.select_neighborhood(i)
                parent1 = self.tournament_selection(neighborhood)
                parent2 = self.tournament_selection(neighborhood)
                offspring = self.reproduction(parent1,parent2)
                fitness_offspring = self.problem.evaluate(offspring)
                if fitness_offspring.objectives[0] < self.fitness_values[i].objectives[0]:
                    self.population[i] = offspring
                    self.fitness_values[i] = fitness_offspring
            self.evaluations += 1
            self.best_fitness_per_epoch.append(self.get_result().objectives[0])
            self.min_variables_per_epoch.append(sum(self.get_result().variables))
            print(f"Epochs: {self.evaluations}/{self.epochs}, Fitness: {self.get_result().objectives[0]}\n")

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
        return self.evaluations >= self.epochs

    def select_neighborhood(self, i):
        neighbors = []
        x,y = i//self.neighborhood_size, i%self.neighborhood_size
        neighbor_index = [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]
        for nx, ny in neighbor_index:
            if 0<=nx<self.neighborhood_size and 0<=ny<self.neighborhood_size:
                neighbors.append(nx*self.neighborhood_size+ny)
        return neighbors

    def tournament_selection(self, neighborhood):
        vecinos = [self.population[i] for i in neighborhood]
        selected = self.selection_operator.execute(vecinos)
        return selected

    def reproduction(self, parent1, parent2):
        offspring = self.crossover_operator.execute([parent1, parent2])
        index_rand = random.randint(0, 1)
        self.mutation_operator.execute(offspring[index_rand])
        return offspring[index_rand]
    
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
    
    def plot_min_variables(self):
        plt.figure()
        plt.plot(self.min_variables_per_epoch)
        plt.xlabel("Epoch")
        plt.ylabel("Min Variables Selected")
        plt.title("Minimum Variables Selected per Epoch")
        plt.show()

    def save_csv(self,filename:str):
        data = {'Best Fitness':self.best_fitness_per_epoch,
                'Min Variables':self.min_variables_per_epoch}
        pd.DataFrame(data).to_csv(filename,index=False)
        print(f"Data saved in {filename}!")

    def get_name(self) -> str:
        return "Cellular Genetic Algorithm"