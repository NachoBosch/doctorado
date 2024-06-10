

function create_initial_solutions(pop_size,gene_lenght):
    population = []
    for i in 0 to pop_size-1:
        individual = []
        for j in 0 to gene_length-1:
            random_int = random.binary(0,1)
            individual.append(random_int)
        population.append(individual)
    return population

function evaluate(individual):
    fitness = fitness_function(individual)

function select_neighborhood(index,population,grid_size):
    neighbors = []
    x, y = index//grid_size, index%grid_size
    neighbor_index = [(X-1,y),(x+1,y),(x,y-1),(x,y+1)] #4 vecinos
    for nx,my in neighbor_index:
        if (nx>=0 & nx<=grid_size) & (ny>=0 & ny<=grid_size):
            neighbors.append(population[nx*grid_size + ny])
    return neighbors

function crossover(parent1,parent2):
    point = random.integer(1,gene_lenght-1)
    child1 = parent1[0:point] + parent2[point:]
    child2 = parent2[0:point] + parent1[point:]
    return child1,child2

function mutation(individual, rate):
    for i in 0 to gene_lenght-1:
        if random.randn() < rate:
            individual[i]=1-individual[i]
    return individual


pop_size = 100
gene_length = 500
grid_size = 10
max_generations = 100
population = create_initial_solutions()
for generation in 0 to max_generations:
    fitness_value = []
    for i in 0 to pop_size-1:
        fitness_value.append(evaluate(population[i]))
    selected_parents = []
    for i in 0 to pop_size-1:
        neighborhood = select_neighborhood(i, population, grid_size)
        parent1 = tournament_selection(neighborhood)
        parent2 = tournament_selection(neighborhood)
        selected_parents.append(parent1,parent2)
    offspring = []
    for parents in selected_parents:
        child1,child2 = crossover(parents[0],parents[1])
        offspring.append(child1)
        offspring.append(child2)
    for i in 0 to length(offspring)-1:
        offspring[i] = mutate(offspring[i],rate)
    population = offspring
    

