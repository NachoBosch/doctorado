import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.join(current_dir.replace('experimentos', ''))
sys.path.append(module_dir)

from jmetal.problems import FeatureSelectionHutington
from jmetal.algorithms.SS_LS import ScatterSearch
from jmetal.core import crossover, mutation, selection
from jmetal.util.termination_criterion import StoppingByEvaluations
from results.results_local import Results
from jmetal.util.observer import PrintObjectivesObserver
from jmetal.util import load

data = load.huntington()
models_names, models = load.models()

#PARAMETERS
params = {
        'population':100,
        'ref_set_size':10,
        'evals' : 1000,
        'mut_p' :0.01,
        'cx_p':0.9,
        'alfa':0.9,
        'encoder':data[3]
        }

#PROBLEM
problem = FeatureSelectionHutington.FeatureSelectionHD(data,params['alfa'],models[3])

#OPERATORS
cros = crossover.SPXCrossover(params['cx_p'])
mut = mutation.BitFlipMutation(params['mut_p'])
criterion = StoppingByEvaluations(params['evals'])

# # ALGORITHM
algorithm = ScatterSearch(
    problem = problem,
    population_size = params['population'],
    reference_set_size = params['ref_set_size'],
    mutation = mut,
    crossover = cros,
    termination_criterion = criterion,
)

algorithm.observable.register(observer=PrintObjectivesObserver(10))
algorithm.run()

# RESULTS
test = 'SS'
Results.results(algorithm,test,data[2],params)
