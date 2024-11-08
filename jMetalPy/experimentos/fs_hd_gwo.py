import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.join(current_dir.replace('experimentos', ''))
sys.path.append(module_dir)

from jmetal.problems import FeatureSelectionHutington
from jmetal.algorithms.GWO import BinaryGWOAlgorithm
from jmetal.algorithms.BOA import BinaryBOA
from jmetal.util.termination_criterion import StoppingByEvaluations
from results.results_local import Results
from jmetal.util.observer import PrintObjectivesObserver
from jmetal.util import load

data = load.huntington()
models_names, models = load.models()
#PARAMETERS
params = {
        'evals' : 1000,
        'alfa':0.9,
        'encoder':data[3]
        }

#PROBLEM
problem = FeatureSelectionHutington.FeatureSelectionHD(data,params['alfa'],models[0])

#OPERATORS
criterion = StoppingByEvaluations(params['evals'])

# # ALGORITHM
# algorithm = BinaryGWOAlgorithm(
#                 problem = problem,
#                 population_size = 100,
#                 max_evaluations = params['evals'],   
#                 termination_criterion = criterion)

algorithm = BinaryBOA(
    problem = problem,
    population_size = 100,
    a = 0.1,
    c = 0.25,
    p = 0.8,
    max_evaluations=params['evals'],
    termination_criterion=criterion
)

algorithm.observable.register(observer=PrintObjectivesObserver(10))
algorithm.run()

# RESULTS
test = 'BOA'
Results.results(algorithm,test,data[2],params)

# algorithm.plot_fitness()
# algorithm.plot_min_variables()
# algorithm.save_csv(f'Results/Resultados_CGA/Resultados_nuevos/{test}.csv')