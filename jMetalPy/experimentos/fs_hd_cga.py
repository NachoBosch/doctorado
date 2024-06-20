import pandas as pd
import numpy as np
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.join(current_dir.replace('experimentos', ''))
sys.path.append(module_dir)

from jmetal.problems import FeatureSelectionHutington
from jmetal.algorithms.cga import CellularGeneticAlgorithm
from jmetal.core import crossover, mutation, selection
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.neighborhood import L5
from jmetal.util.update_policy import LineSweep
from results.results_local import Results
from jmetal.util.observer import PrintObjectivesObserver
from jmetal.util import load
from sklearn.preprocessing import LabelEncoder,MinMaxScaler



data = load.huntington()
models = load.models()

#PARAMETERS
params = {'pobl': 100,
        'off_pobl': 100,
        'evals' : 10000,
        'mut_p' :0.01,
        'cross_p': 0.9,
        'alfa':0.8,
        'encoder':data[3]
        }

#PROBLEM
problem = FeatureSelectionHutington.FeatureSelectionHD(data,params['alfa'],models[1])

#OPERATORS
mut = mutation.BitFlipMutation(params['mut_p'])
cross = crossover.SPXCrossover(params['cross_p'])
selection = selection.BinaryTournamentSelection()
criterion = StoppingByEvaluations(params['evals'])

# # ALGORITHM
algorithm = CellularGeneticAlgorithm(
    problem = problem,
    pop_size = params['pobl'],
    mutation = mut,
    crossover = cross,
    selection = selection,
    termination_criterion = criterion,
    neighborhood=L5(rows=10,columns=10),
    cell_update_policy=LineSweep()
)

algorithm.observable.register(observer=PrintObjectivesObserver(100))
algorithm.run()

# RESULTS
test = 'KNN'
Results.results(algorithm,test,data[2],params)

# algorithm.plot_fitness()
# algorithm.plot_min_variables()
# algorithm.save_csv(f'Results/Resultados_CGA/Resultados_nuevos/{test}.csv')