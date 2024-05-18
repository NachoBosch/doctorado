import numpy as np
from mealpy.optimizer import Optimizer

class CellularGA(Optimizer):
    def __init__(self, epoch: int = 10000,
                 pop_size: int = 100,
                 pc: float = 0.95,
                 pm: float = 0.025,
                 grid_shape = (10, 10),
                 selection = "tournament",
                 k_way = 0.2,
                 crossover = "one_point",
                 mutation = "flip") -> None:
        super().__init__()

        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.pc = self.validator.check_float("pc", pc, (0, 1.0))
        self.pm = self.validator.check_float("pm", pm, (0, 1.0))
        self.grid_shape = grid_shape
        self.grid = self.create_grid()
        self.selection = selection
        self.k_way = k_way
        self.crossover = crossover
        self.mutation = mutation

    def create_grid(self):
        assert self.pop_size == self.grid_shape[0] * self.grid_shape[1], "Population size must match grid shape."
        grid = np.arange(self.pop_size).reshape(self.grid_shape)
        return grid

    def get_neighbors(self, idx):
        row, col = divmod(idx, self.grid_shape[1])
        neighbors = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
        neighbors = [(r % self.grid_shape[0], c % self.grid_shape[1]) for r, c in neighbors]
        return [self.grid[r, c] for r, c in neighbors]

    def selection_process__(self, idx, list_fitness):
        neighbors = self.get_neighbors(idx)
        if len(neighbors) < 2:
            print("Not enough neighbors for selection. Using random selection.")
            id_c1, id_c2 = self.generator.choice(range(self.pop_size), 2, replace=False)
        else:
            id_c1, id_c2 = self.get_index_kway_tournament_selection(neighbors, list_fitness, k_way=self.k_way, output=2)
        return self.pop[id_c1].solution, self.pop[id_c2].solution

    def crossover_process__(self, dad, mom):
        cut = self.generator.integers(1, self.problem.n_dims-1)
        w1 = np.concatenate([dad[:cut], mom[cut:]])
        w2 = np.concatenate([mom[:cut], dad[cut:]])
        return w1, w2

    def mutation_process__(self, child):
        idx = self.generator.integers(0, self.problem.n_dims)
        child[idx] = self.generator.uniform(self.problem.lb[idx], self.problem.ub[idx])
        return child

    def survivor_process__(self, pop, pop_child, list_fitness):
        pop_new = []
        for idx in range(0, self.pop_size):
            neighbors = self.get_neighbors(idx)
            if len(neighbors) < 1:
                print("Not enough neighbors for selection. Using random selection.")
                id_child = self.generator.choice(range(self.pop_size))
            else:
                id_child = self.get_index_kway_tournament_selection(neighbors, list_fitness, k_way=max(2, 0.1*len(neighbors)), output=1, reverse=True)[0]
            pop_new.append(self.get_better_agent(pop_child[idx], pop[id_child], self.problem.minmax))
        return pop_new
    
    def evolve(self, epoch):
        list_fitness = np.array([agent.target.fitness for agent in self.pop])
        # print("List fitness", list_fitness)
        pop_new = []
        for i in range(self.pop_size):
            child1, child2 = self.selection_process__(i, list_fitness)
            if self.generator.random() < self.pc:
                child1, child2 = self.crossover_process__(child1, child2)

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

        self.pop = self.survivor_process__(self.pop, pop_new, list_fitness)

    def get_index_kway_tournament_selection(self, pop: [] = None, list_fitness: [] = None, k_way: float = 0.2, output: int = 2, reverse: bool = False) -> []:
        if 0 < k_way < 1:
            k_way = int(k_way * len(pop))
        k_way = max(2, k_way)  # Asegurarse de que k_way sea al menos 2
        list_id = self.generator.choice(range(len(pop)), k_way, replace=False)
        list_parents = [[idx, list_fitness[pop[idx]]] for idx in list_id]
        if self.problem.minmax == "min":
            list_parents = sorted(list_parents, key=lambda agent: agent[1])
        else:
            list_parents = sorted(list_parents, key=lambda agent: agent[1], reverse=True)
        if reverse:
            return [parent[0] for parent in list_parents[-output:]]
        return [parent[0] for parent in list_parents[:output]]