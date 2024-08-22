import copy
from typing import TypeVar, List
import numpy as np

from jmetal.config import store
from jmetal.core.algorithm import ParticleSwarmOptimization
from jmetal.core.problem import Problem
from jmetal.util.comparator import Comparator
from jmetal.util.evaluator import Evaluator
from jmetal.util.generator import Generator
from jmetal.util.termination_criterion import TerminationCriterion
from jmetal.core.solution import BinarySolution

class BinaryPSOAlgorithm(ParticleSwarmOptimization):
    def __init__(self,
                 problem: Problem,
                 swarm_size: int,
                 inertia_weight: float,
                 cognitive_coefficient: float,
                 social_coefficient: float,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 dominance_comparator: Comparator = store.default_comparator):
        
        super(BinaryPSOAlgorithm, self).__init__(problem=problem, swarm_size=swarm_size)

        self.inertia_weight = inertia_weight
        self.cognitive_coefficient = cognitive_coefficient
        self.social_coefficient = social_coefficient

        self.dominance_comparator = dominance_comparator
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)

        self.velocities = None
        self.pbests = None
        self.gbest = None
        self.gbest_fitness = float('inf')

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def create_initial_solutions(self) -> List[BinarySolution]:
        return [self.problem.create_solution() for _ in range(self.swarm_size)]

    def evaluate(self, swarm: List[BinarySolution]):
        return self.population_evaluator.evaluate(swarm, self.problem)

    def initialize_velocity(self, swarm: List[BinarySolution]):
        self.velocities = [np.zeros(len(particle.variables)) for particle in swarm]

    def initialize_particle_best(self, swarm: List[BinarySolution]):
        self.pbests = swarm.copy()

    def initialize_global_best(self, swarm: List[BinarySolution]):
        self.gbest = min(swarm, key=lambda s: s.objectives[0])
        self.gbest_fitness = self.gbest.objectives[0]

    def update_velocity(self, swarm: List[BinarySolution]):
        for i, particle in enumerate(swarm):
            r1, r2 = np.random.random(), np.random.random()
            cognitive_velocity = self.cognitive_coefficient * r1 * (np.sum(self.pbests[i].variables) - np.sum(particle.variables))
            social_velocity = self.social_coefficient * r2 * (np.sum(self.gbest.variables) - np.sum(particle.variables))
            self.velocities[i] = self.inertia_weight * self.velocities[i] + cognitive_velocity + social_velocity

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def update_position(self, swarm: List[BinarySolution]):
        for i, particle in enumerate(swarm):
            s = self.sigmoid(self.velocities[i])
            particle.variables = [True if np.random.random() < s[j] else False for j in range(len(s))]

    def update_particle_best(self, swarm: List[BinarySolution]):
        for i, particle in enumerate(swarm):
            if self.dominance_comparator.compare(particle, self.pbests[i]) < 0:
                self.pbests[i] = particle

    def update_global_best(self, swarm: List[BinarySolution]):
        for particle in swarm:
            if self.dominance_comparator.compare(particle, self.gbest) < 0:
                self.gbest = particle
                self.gbest_fitness = particle.objectives[0]

    def perturbation(self, swarm: List[BinarySolution]):
        pass

    def update_progress(self):
        self.evaluations += self.swarm_size
        print(f"Evaluations: {self.evaluations}")
        observable_data = self.observable_data()
        self.observable.notify_all(**observable_data)

    def result(self):
        return self.gbest

    def get_name(self) -> str:
        return 'Binary Particle Swarm Optimization Algorithm'
