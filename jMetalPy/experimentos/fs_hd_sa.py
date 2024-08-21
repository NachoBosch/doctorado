import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.join(current_dir.replace('experimentos', ''))
sys.path.append(module_dir)

from jmetal.problems import FeatureSelectionHutington
from jmetal.algorithms.simulated_annealing import SimulatedAnnealing
from jmetal.core import crossover, mutation, selection
from jmetal.util.termination_criterion import StoppingByEvaluations
from results.results_local import Results
from jmetal.util.observer import PrintObjectivesObserver
from jmetal.util import load

data = load.huntington()
models = load.models()

#PARAMETERS
params = {
        'evals' : 1000,
        'mut_p' :0.01,
        'alfa':0.9,
        'encoder':data[3]
        }

#PROBLEM
problem = FeatureSelectionHutington.FeatureSelectionHD(data,params['alfa'],models[3])

#OPERATORS
mut = mutation.BitFlipMutation(params['mut_p'])
criterion = StoppingByEvaluations(params['evals'])

# # ALGORITHM
algorithm = SimulatedAnnealing(
    problem = problem,
    mutation = mut,
    termination_criterion = criterion
)

algorithm.observable.register(observer=PrintObjectivesObserver(10))
algorithm.run()

# RESULTS
test = 'AB'
Results.results(algorithm,test,data[2],params)

# algorithm.plot_fitness()
# algorithm.plot_min_variables()
# algorithm.save_csv(f'Results/Resultados_CGA/Resultados_nuevos/{test}.csv')