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
                ramp_angle: float = 30,      # Ángulo de la rampa en grados
                friction_coefficient: float = 0.1,  # Coeficiente de fricción
                gravity: float = 9.81,             # Gravedad
                termination_criterion: TerminationCriterion = store.default_termination_criteria,        
                population_generator: Generator = store.default_generator,
                population_evaluator: Evaluator = store.default_evaluator,
                dominance_comparator: Comparator = store.default_comparator,):
        
        super().__init__()
        self.problem = problem
        self.population_size = population_size
        self.ramp_angle = np.radians(ramp_angle)  # Convertimos el ángulo a radianes
        self.initial_angle = self.ramp_angle
        self.friction_coefficient = friction_coefficient
        self.gravity = gravity
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.dominance_comparator = dominance_comparator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.population = []
        self.best_solution = None
        self.evaluations = 0
        self.previous_velocities = None
        self.stagnation_counter = 0
        self.momentum_coefficient: float = 0.7
        self.adaptive_angle: bool = True
        self.best_fitness_history = []
        self.previous_velocities = np.zeros((population_size, problem.number_of_variables))

    def create_initial_solutions(self):
        print("Primer llamado")
        self.population = [self.problem.create_solution() for _ in range(self.population_size)]
        return self.population

    def evaluate(self, solution_list):
        print("Segundo llamado", len(solution_list))
        for solution in solution_list:
            self.problem.evaluate(solution)
        return solution_list
    
    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def init_progress(self):
        print("Tercer llamado")
        self.best_solution = min(self.population, key=lambda sol: sol.objectives[0])
        print(f"Init Best: {self.best_solution.objectives[0]}")
        self.evaluations = self.population_size

    def calculate_force_effect(self):
        max_angle_increase = np.radians(45)
        if self.adaptive_angle and len(self.best_fitness_history) > 10:
            if len(set(self.best_fitness_history[-5:])) == 1:
                self.stagnation_counter += 1
                self.ramp_angle = min(self.initial_angle * (1 + 0.1 * self.stagnation_counter), max_angle_increase)
            else:
                self.stagnation_counter = 0
                self.ramp_angle = self.initial_angle

        ramp_force = self.gravity * np.sin(self.ramp_angle)
        friction_force = self.friction_coefficient * self.gravity * np.cos(self.ramp_angle)
        net_force = max(min(ramp_force - friction_force, 1), -1)
        return net_force

    def v_shape_transfer(self, x):
        return abs(np.tanh(x))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def update_binary_position(self, probability):
        return True if random.random() < probability else False

    def update_positions(self):
        force_effect = self.calculate_force_effect()
        for idx, agent in enumerate(self.population):
            current_velocities = np.zeros(self.problem.number_of_variables)
            for i in range(self.problem.number_of_variables):
                new_velocity = (self.momentum_coefficient * self.previous_velocities[idx][i] + 
                              (1 - self.momentum_coefficient) * random.uniform(-1, 1) * force_effect)
                current_velocities[i] = new_velocity
                prob = self.v_shape_transfer(new_velocity)
                # prob = self.sigmoid(new_velocity)
                agent.variables[i] = self.update_binary_position(prob)
            self.previous_velocities[idx] = current_velocities

    def update_best_solution(self):
        current_best = min(self.population, key=lambda sol: sol.objectives[0])
        if current_best.objectives[0] < self.best_solution.objectives[0]:
            self.best_solution = copy.deepcopy(current_best)
            print(self.best_solution.objectives[0])
        self.best_fitness_history.append(self.best_solution.objectives[0])
        
    def step(self):
        self.update_positions()
        self.evaluate(self.population)
        self.update_best_solution()

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
        print("RESULT")
        return self.best_solution

    def get_name(self):
        return "Binary Giza Pyramids Construction Algorithm"