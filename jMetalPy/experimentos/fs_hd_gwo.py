import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.join(current_dir.replace('experimentos', ''))
sys.path.append(module_dir)

from jmetal.problems import FeatureSelectionHutington
from jmetal.algorithms.GWO import BinaryGWOAlgorithm
from jmetal.algorithms.BOA import BinaryBOA
from jmetal.algorithms.uGA import MicroGeneticAlgorithm
from jmetal.algorithms.GPC_2 import BinaryGPC
from jmetal.core import crossover, mutation, selection
from jmetal.util.termination_criterion import StoppingByEvaluations
from results.results_local import Results
from jmetal.util.observer import PrintObjectivesObserver
from jmetal.util import load

data = load.huntington()
models_names, models = load.models()
#PARAMETERS
params = {
        'evals' : 5000,
        'alfa':0.9,
        'encoder':data[3]
        }

#PROBLEM
problem = FeatureSelectionHutington.FeatureSelectionHD(data,params['alfa'],models[2])

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
    c = 0.3,
    p = 0.8,
    max_evaluations=params['evals'],
    termination_criterion=criterion
)

# algorithm = MicroGeneticAlgorithm(
#                         problem = problem,
#                         population_size= 10,
#                         mutation = mutation.BitFlipMutation(0.01),
#                         crossover = crossover.SPXCrossover(0.9),
#                         selection = selection.BinaryTournamentSelection(),
#                         reinicio = 50,
#                         termination_criterion=criterion)

# algorithm = BinaryGPC(problem = problem,
#                 population_size= 100,
#                 #ramp_angle = 30 , 
#                 #friction_coefficient= 0.1,     # Factor de fricción, simula resistencia
#                 #gravity= 9.8,      # Factor de gravedad
#                 termination_criterion=criterion)

algorithm.observable.register(observer=PrintObjectivesObserver(10))
algorithm.run()

# RESULTS
test = 'GPC'
Results.results(algorithm,test,data[2],params)

# algorithm.plot_fitness()
# algorithm.plot_min_variables()
# algorithm.save_csv(f'Results/Resultados_CGA/Resultados_nuevos/{test}.csv')