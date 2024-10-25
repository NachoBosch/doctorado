import sys
import os
import pandas as pd
current_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.join(current_dir.replace('experimentos', ''))
sys.path.append(module_dir)

from jmetal.problems import Biclustering
from jmetal.algorithms.DE import DE
from jmetal.util.termination_criterion import StoppingByEvaluations
from results.results_local import Results
from jmetal.util.observer import PrintObjectivesObserver
from jmetal.util import load

data = load.huntington_bic()

# print(data)

#PARAMETERS
params = {
        'population':20,
        'evals' : 1000,
        'CR' :0.5,
        'F':0.3,
        'alfa':0.9
        }

#PROBLEM
problem = Biclustering.BiclusteringProblem(data,params['alfa'])

#OPERATORS
criterion = StoppingByEvaluations(params['evals'])

# # ALGORITHM
algorithm = DE(
    problem = problem,
    population_size=params['population'],
    CR=params['CR'],
    F=params['F'],
    termination_criterion = criterion
)

algorithm.observable.register(observer=PrintObjectivesObserver(10))
algorithm.run()

# RESULTS
test = 'Biclustering'
# Results.results(algorithm,test,data,params)

# algorithm.plot_fitness()
# algorithm.plot_min_variables()
# algorithm.save_csv(f'Results/Resultados_CGA/Resultados_nuevos/{test}.csv')