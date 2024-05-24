import pandas as pd

from jmetal.algorithm.singleobjective import GeneticAlgorithm

from Problems import GA
from Results import Results
from Core import crossover, mutation, selection

from sklearn.preprocessing import LabelEncoder



#DATA
df_hd = pd.read_csv('D:/Doctorado/doctorado/Data/HD_filtered.csv')

#PRE-SETS
encoder = LabelEncoder()
X = df_hd.drop(columns=['Samples','Grade']).to_numpy()
y = encoder.fit_transform(df_hd.Grade.to_numpy())
clases = list(df_hd.columns[:-2])

#PARAMETERS
params = {'pobl': 100,
        'off_pobl': 100,
        'evals' : 1000,
        'mut_p' :0.1,
        'cross_p': 0.8,
        'alfa':0.3,
        'encoder':encoder
        }

#PROBLEM
problem = GA.FeatureSelectionGA(X, y, params['alfa'])

#OPERATORS
mut = mutation.BitFlipMutation(params['mut_p'])
cross = crossover.SPXCrossover(params['cross_p'])
selection = selection.BinaryTournamentSelection()

# # ALGORITHM
algorithm = GeneticAlgorithm(
    problem=problem,
    population_size=params['pobl'],
    offspring_population_size=params['off_pobl'],
    mutation=mut,
    crossover=cross,
    selection=selection,
    termination_criterion=criterion
)

algorithm.run()

# RESULTS
experiment = 'Experimento1'
test = str(9)
Results.results(algorithm,experiment,test,clases,params)

# algorithm.plot_fitness()
