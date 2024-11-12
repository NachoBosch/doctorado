import numpy as np
from jmetal.core.problem import Problem
from jmetal.core.solution import BinarySolution
from jmetal.core.algorithm import Algorithm
from jmetal.config import store
from jmetal.util.termination_criterion import TerminationCriterion
from typing import List
from jmetal.util.comparator import Comparator
from jmetal.util.evaluator import Evaluator
from jmetal.util.generator import Generator
import random
import time
import copy

class BinaryGPC(Algorithm[BinarySolution, BinarySolution]):
    def __init__(self, 
                problem: Problem,
                population_size: int,
                termination_criterion: TerminationCriterion = store.default_termination_criteria,        
                population_generator: Generator = store.default_generator,
                population_evaluator: Evaluator = store.default_evaluator,
                dominance_comparator: Comparator = store.default_comparator):
        super().__init__()
        self.problem = problem
        self.population_size = population_size
        self.best_solution = None
        self.population = []
        # Parámetros físicos
        self.g = 9.81  # gravedad
        self.theta = np.radians(14)  # ángulo de la rampa (30 grados)
        self.mu_k = 0.1  # coeficiente de fricción cinética
        self.mass = 1.0  # masa normalizada del bloque
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.dominance_comparator = dominance_comparator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)

    def calculate_displacement(self):
        v0 = np.random.uniform(0, 1, self.problem.number_of_variables)
        # Fuerza neta = mg sin(θ) - µk mg cos(θ)
        F_net = (self.mass * self.g * np.sin(self.theta) - 
                self.mu_k * self.mass * self.g * np.cos(self.theta))
        a = F_net / self.mass  # aceleración
        d = (v0 ** 2) / (2 * abs(a))  # desplazamiento
        return d

    def calculate_worker_movement(self):
        F_worker = self.mass * self.g * np.sin(self.theta)  # Fuerza del trabajador
        movement = F_worker / (self.mass * self.g)  # Movimiento normalizado
        return movement * np.random.uniform(-1, 1, self.problem.number_of_variables)

    def estimate_new_position(self, movement, displacement):
        return movement + displacement

    def worker_substitution_probability(self):
        energy = 0.5 * self.mass * self.g * np.sin(self.theta)
        return np.clip(energy / (self.mass * self.g), 0, 1)

    def transfer_function(self, x):
        # return 1 / (1 + np.exp(-2 * x))
        return abs(np.tanh(x))

    def create_initial_solutions(self):
        self.population = [self.problem.create_solution() 
                         for _ in range(self.population_size)]
        return self.population

    def evaluate(self, solution_list):
        for solution in solution_list:
            self.problem.evaluate(solution)
        return solution_list

    def step(self):
        for i in range(self.population_size):
            # Calcular desplazamiento físico de los bloques
            d = self.calculate_displacement()
            # Calcular movimiento de los trabajadores
            x = self.calculate_worker_movement()
            # Estimar nueva posición
            new_position = self.estimate_new_position(x, d)
            # Probabilidad de sustitución de trabajadores
            prob_substitution = self.worker_substitution_probability()
            # Transformación binaria usando función S-shape
            prob = self.transfer_function(new_position)
            # Crear nueva solución binaria
            new_solution = self.problem.create_solution()
            for j in range(self.problem.number_of_variables):
                if np.random.random() < prob[j]:
                    new_solution.variables[j] = True
                else:
                    new_solution.variables[j] = False

            self.problem.evaluate(new_solution)
            
            if np.random.random() < prob_substitution:
                if (not self.best_solution or 
                    new_solution.objectives[0] < self.population[i].objectives[0]):
                    self.population[i] = copy.deepcopy(new_solution)
                    
                    # Actualizar mejor solución global si corresponde
                    if (not self.best_solution or 
                        new_solution.objectives[0] < self.best_solution.objectives[0]):
                        self.best_solution = copy.deepcopy(new_solution)

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def init_progress(self):
        self.best_solution = min(self.population, key=lambda sol: sol.objectives[0])
        self.evaluations = self.population_size

    def update_progress(self):
        self.evaluations += self.population_size
        observable_data = self.observable_data()
        self.observable.notify_all(**observable_data)

    def observable_data(self):
        return {
            "PROBLEM": self.problem,
            "EVALUATIONS": self.evaluations,
            "SOLUTIONS": self.result(),
            "COMPUTING_TIME": time.time() - self.start_computing_time,
        }

    def result(self):
        return self.best_solution

    def get_name(self):
        return "Binary Giza Pyramids Construction Algorithm"