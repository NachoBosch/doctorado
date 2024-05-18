import numpy as np
from mealpy.optimizer import Optimizer

class BaseGA(Optimizer):
    """
    The original version of: Genetic Algorithm (GA)
    """

    def __init__(self, epoch: int = 10000, 
                 pop_size: int = 100, 
                 pc: float = 0.95, 
                 pm: float = 0.025, 
                 sort_flag = False,
                 selection = "tournament",
                 k_way = 0.2,
                 crossover = "one_point",
                 mutation = "flip") -> None:
        super().__init__()

        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.pc = self.validator.check_float("pc", pc, (0, 1.0))
        self.pm = self.validator.check_float("pm", pm, (0, 1.0))
        self.sort_flag = sort_flag
        self.selection = selection
        self.k_way = k_way
        self.crossover = crossover
        self.mutation = mutation

    def selection_process__(self, list_fitness):
        ## tournament
        id_c1, id_c2 = self.get_index_kway_tournament_selection(self.pop, k_way=self.k_way, output=2)
        return self.pop[id_c1].solution, self.pop[id_c2].solution

    def crossover_process__(self, dad, mom):
        cut = self.generator.integers(1, self.problem.n_dims-1)
        w1 = np.concatenate([dad[:cut], mom[cut:]])
        w2 = np.concatenate([mom[:cut], dad[cut:]])
        return w1, w2

    def mutation_process__(self, child):
        # "flip"
        idx = self.generator.integers(0, self.problem.n_dims)
        child[idx] = self.generator.uniform(self.problem.lb[idx], self.problem.ub[idx])
        return child

    def survivor_process__(self, pop, pop_child):
        pop_new = []
        for idx in range(0, self.pop_size):
            id_child = self.get_index_kway_tournament_selection(pop, k_way=0.1, output=1, reverse=True)[0]
            pop_new.append(self.get_better_agent(pop_child[idx], pop[id_child], self.problem.minmax))
        return pop_new
    
    def evolve(self,epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class
        """
        list_fitness = np.array([agent.target.fitness for agent in self.pop])
        print("List fitness",list_fitness)
        pop_new = []
        for i in range(0, int(self.pop_size/2)):
            ### Selection
            child1, child2 = self.selection_process__(list_fitness)
            ### Crossover
            if self.generator.random() < self.pc:
                child1, child2 = self.crossover_process__(child1, child2)

            ### Mutation
            child1 = self.mutation_process__(child1)
            child2 = self.mutation_process__(child2)

            child1 = self.correct_solution(child1)
            child2 = self.correct_solution(child2)

            agent1 = self.generate_empty_agent(child1)
            agent2 = self.generate_empty_agent(child2)

            pop_new.append(agent1)
            pop_new.append(agent2)

            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-2].target = self.get_target(child1)
                pop_new[-1].target = self.get_target(child2)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
        ### Survivor Selection
        self.pop = self.survivor_process__(self.pop, pop_new)