import matplotlib.pyplot as plt
import pandas as pd
import random

class GeneticAlgorithm():
    def __init__(
            self,
            problem,
            population_size: int,
            offspring_population_size: int,
            max_evaluations: int,
            mutation,
            crossover,
            selection,
            local_search_probability=0.2):
        
        self.problem = problem
        self.population_size = population_size
        self.offspring_population_size = offspring_population_size
        self.mutation_operator = mutation
        self.crossover_operator = crossover
        self.selection_operator = selection
        self.max_evaluations = max_evaluations
        self.evaluations = 0
        self.best_fitness_per_epoch = []
        self.best_acc_per_epoch=[]
        self.min_variables_per_epoch = []
        self.epochs = self.max_evaluations // self.population_size
        self.local_search_probability = local_search_probability

        self.mating_pool_size = (
            self.offspring_population_size
            * self.crossover_operator.get_number_of_parents()
            // self.crossover_operator.get_number_of_children()
        )

        if self.mating_pool_size < self.crossover_operator.get_number_of_children():
            self.mating_pool_size = self.crossover_operator.get_number_of_children()

    def create_initial_solutions(self) -> []:
        return [self.problem.create_solution() for _ in range(self.population_size)]

    def evaluate(self, population: []):
        for solution in population:
            self.problem.evaluate(solution)
        self.evaluations += 1
        return population

    def stopping_condition_is_met(self) -> bool:
        return self.evaluations >= self.epochs

    def selection(self, population: []):
        return [self.selection_operator.execute(population) for _ in range(self.mating_pool_size)]

    def reproduction(self, mating_population):
        number_of_parents_to_combine = self.crossover_operator.get_number_of_parents()
        offspring_population = []
        for i in range(0, self.offspring_population_size, number_of_parents_to_combine):
            parents = mating_population[i:i + number_of_parents_to_combine]
            offspring = self.crossover_operator.execute(parents)
            for solution in offspring:
                self.mutation_operator.execute(solution)
                offspring_population.append(solution)
                if len(offspring_population) >= self.offspring_population_size:
                    break
        return offspring_population

    def replacement(self, population: [], offspring_population: []) -> []:
        combined = population + offspring_population
        combined.sort(key=lambda s: s.objectives[0])
        return combined[:self.population_size]
    
    def local_search(self, solution):
        """ Aplica una búsqueda local simple (Hill Climbing) en la solución."""
        if random.random() < self.local_search_probability:  # Solo aplica en algunos casos
            for _ in range(5):  # Número de iteraciones de mejora
                neighbor = self.problem.create_neighbor(solution)
                self.problem.evaluate(neighbor)
                if neighbor.objectives[0] < solution.objectives[0]:
                    solution.variables = neighbor.variables[:]  # Actualiza si es mejor
                    solution.objectives = neighbor.objectives[:]
        return solution
    
    def run(self):
        self.population = self.create_initial_solutions()
        self.evaluate(self.population)
        while not self.stopping_condition_is_met():
            mating_population = self.selection(self.population)
            offspring_population = self.reproduction(mating_population)
            offspring_population = self.evaluate(offspring_population)
            
            # Aplicar búsqueda local a algunos individuos
            offspring_population = [self.local_search(sol) for sol in offspring_population]
            
            self.population = self.replacement(self.population, offspring_population)
            best_fitness = min(solution.objectives[0] for solution in self.population)
            best_acc = min(solution.objectives[1] for solution in self.population)
            self.best_fitness_per_epoch.append(best_fitness)
            self.min_variables_per_epoch.append(min(sum(solution.variables) for solution in self.population))
            print(f"Epochs: {self.evaluations}/{self.epochs}, Fitness: {self.get_result().objectives[0]}")

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

    def save_csv(self, filename: str):
        data = {'Best Fitness': self.best_fitness_per_epoch,
                'Min Variables': self.min_variables_per_epoch}
        pd.DataFrame(data).to_csv(filename, index=False)
        print(f"Data saved in {filename}!")

    def get_name(self) -> str:
        return "Memetic Algorithm"
