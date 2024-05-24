import pandas as pd
import numpy as np

from Problems import CellularGA
from Algorithms.CGA import CellularGeneticAlgorithm
from Core import crossover, mutation, selection

from sklearn.preprocessing import LabelEncoder

from Results import Results

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
        'evals' : 10000,
        'mut_p' :0.1,
        'cross_p': 0.8,
        'alfa':0.3,
        'encoder':encoder
        }

#PROBLEM
problem = CellularGA.FeatureSelectionCGA(X, y, params['alfa'])

#OPERATORS
mut = mutation.BitFlipMutation(params['mut_p'])
cross = crossover.SPXCrossover(params['cross_p'])
selection = selection.BinaryTournamentSelection()

# # ALGORITHM
algorithm = CellularGeneticAlgorithm(
    problem = problem,
    population_size = params['pobl'],
    offspring_population_size = params['off_pobl'],
    max_evaluations = params['evals'],
    mutation = mut,
    crossover = cross,
    selection = selection,
    neighborhood_size = 10
)

algorithm.run()

# RESULTS
experiment = 'Experimento1'
test = str(9)
Results.results(algorithm,experiment,test,clases,params)

algorithm.plot_fitness()