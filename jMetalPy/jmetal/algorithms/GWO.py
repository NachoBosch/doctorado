import numpy as np
from jmetal.core.solution import BinarySolution
from jmetal.core.problem import Problem
from jmetal.core.algorithm import Algorithm
from jmetal.util.termination_criterion import TerminationCriterion
from typing import List
import random
import time
from jmetal.config import store

class BinaryGWOAlgorithm(Algorithm[BinarySolution, BinarySolution]):
    def __init__(self, 
                 problem: Problem, 
                 population_size: int, 
                 max_evaluations: int,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria):
        super().__init__()
        self.problem = problem
        self.population_size = population_size
        self.max_evaluations = max_evaluations
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.solutions = []
        self.alpha, self.beta, self.delta = None, None, None  # Los 3 mejores lobos
        self.evaluations = 0

    def create_initial_solutions(self) -> List[BinarySolution]:
        return [self.problem.create_solution() for _ in range(self.population_size)]

    def evaluate(self, solution_list: List[BinarySolution]) -> List[BinarySolution]:
        return [self.problem.evaluate(solution) for solution in solution_list]

    def init_progress(self) -> None:
        self.solutions = self.create_initial_solutions()
        self.solutions = self.evaluate(self.solutions)
        self.update_leaders()

    def update_leaders(self):
        # Ordenar soluciones por fitness (minimización)
        sorted_solutions = sorted(self.solutions, key=lambda sol: sol.objectives[0])
        self.alpha = sorted_solutions[0]
        self.beta = sorted_solutions[1]
        self.delta = sorted_solutions[2]

    def sigmoid(self, x: float) -> float:
        """Función sigmoide mejorada con factor de escala"""
        return 1 / (1 + np.exp(-2 * x))  # Factor de escala 2 para hacer la función más pronunciada

    def v_shape(self, x: float) -> float:
        """Función de transferencia V-shape"""
        return abs(np.tanh(x))

    def update_binary_position(self, position: float) -> bool:
        """
        Actualización de posición binaria usando una combinación de S-shape y V-shape
        """
        s_value = self.sigmoid(position)
        v_value = self.v_shape(position)
        
        if random.random() < 0.5:  # Alternar entre S-shape y V-shape
            return random.random() < s_value
        else:
            return random.random() < v_value

    def update_positions(self):
        a = 2 * (1 - self.evaluations / self.max_evaluations)  # Control de exploración/explotación
        
        for wolf in self.solutions:
            A1 = 2 * a * random.random() - a  # Coeficiente de caza
            A2 = 2 * a * random.random() - a
            A3 = 2 * a * random.random() - a
            
            C1 = 2 * random.random()  # Coeficiente de énfasis
            C2 = 2 * random.random()
            C3 = 2 * random.random()
            
            for i in range(len(wolf.variables)):
                # Calcular las distancias usando XOR
                d_alpha = C1 * (self.alpha.variables[i] ^ wolf.variables[i])
                d_beta = C2 * (self.beta.variables[i] ^ wolf.variables[i])
                d_delta = C3 * (self.delta.variables[i] ^ wolf.variables[i])
                
                # Calcular nuevas posiciones
                x1 = A1 * d_alpha
                x2 = A2 * d_beta
                x3 = A3 * d_delta
                
                # Promedio ponderado de las posiciones
                x_new = (x1 + x2 + x3) / 3.0
                
                # Actualizar posición usando la función de transferencia
                wolf.variables[i] = self.update_binary_position(x_new)

    def step(self) -> None:
        self.update_positions()
        self.solutions = self.evaluate(self.solutions)
        self.evaluations += len(self.solutions)
        self.update_leaders()
        self.update_progress()

    def stopping_condition_is_met(self) -> bool:
        if self.termination_criterion:
            return self.termination_criterion.is_met
        return self.evaluations >= self.max_evaluations

    def update_progress(self) -> None:
        self.evaluations += self.population_size
        print(f"Evaluations: {self.evaluations}/{self.max_evaluations}")
        observable_data = self.observable_data()
        self.observable.notify_all(**observable_data)

    def observable_data(self) -> dict:
        return {
            "PROBLEM": self.problem,
            "EVALUATIONS": self.evaluations,
            "SOLUTIONS": self.result(),
            "COMPUTING_TIME": time.time() - self.start_computing_time,
        }

    def result(self) -> BinarySolution:
        return self.alpha  # Mejor solución encontrada

    def get_name(self) -> str:
        return "Binary Grey Wolf Optimizer Algorithm"
