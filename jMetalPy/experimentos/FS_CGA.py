import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.join(current_dir.replace('experimentos', ''))
sys.path.append(module_dir)

from jmetal.lab.experiment import Experiment, Job, generate_summary_from_experiment,generate_latex_tables
from jmetal.core.quality_indicator import *
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util import load
from jmetal.problems import FeatureSelectionHutington as fsh
from jmetal.algorithms.cga import CellularGeneticAlgorithm
from jmetal.core import crossover, mutation, selection
from jmetal.util.neighborhood import L5
from jmetal.util.update_policy import LineSweep
import logging
# import cProfile
# import pstats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_dir(path,model_name,alfa):
    # model_name = str(model).replace('()','')
    path = path+model_name+'/alfa_'+str(alfa)
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Creating dir: {path}")
        return path
    else:
        print("Directory is already exists!")
        return path
    


def configure_experiment(problems: dict,n_run: int):
    jobs = []
    for run in range(n_run):
        for problem_tag, problem in problems.items():
            jobs.append(
                Job(
                algorithm=CellularGeneticAlgorithm(
                        problem = problem,
                        pop_size = 100,
                        mutation = mutation.BitFlipMutation(0.01),
                        crossover = crossover.SPXCrossover(0.9),
                        selection = selection.BinaryTournamentSelection(),
                        termination_criterion=StoppingByEvaluations(10000),
                        neighborhood=L5(rows=10,columns=10),
                        cell_update_policy=LineSweep()
                    ),
                algorithm_tag="CX_09",
                problem_tag=problem_tag,
                run=run)
            )
            jobs.append(
                Job(
                algorithm=CellularGeneticAlgorithm(
                        problem = problem,
                        pop_size = 100,
                        mutation = mutation.BitFlipMutation(0.01),
                        crossover = crossover.SPXCrossover(0.8),
                        selection = selection.BinaryTournamentSelection(),
                        termination_criterion=StoppingByEvaluations(10000),
                        neighborhood=L5(rows=10,columns=10),
                        cell_update_policy=LineSweep()
                    ),
                algorithm_tag="CX_08",
                problem_tag=problem_tag,
                run=run)
            )
            jobs.append(
                Job(
                algorithm=CellularGeneticAlgorithm(
                        problem = problem,
                        pop_size = 100,
                        mutation = mutation.BitFlipMutation(0.01),
                        crossover = crossover.SPXCrossover(0.7),
                        selection = selection.BinaryTournamentSelection(),
                        termination_criterion=StoppingByEvaluations(10000),
                        neighborhood=L5(rows=10,columns=10),
                        cell_update_policy=LineSweep()
                    ),
                algorithm_tag="CX_07",
                problem_tag=problem_tag,
                run=run)
            )

    return jobs


data = load.huntington()
alfa = [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
models = load.models()

for model in models[:1]:
    model_name='RF'
    for a in alfa:
        jobs = configure_experiment(problems={"FS_CGA": fsh.FeatureSelectionHD(data,a,model)},
                                    n_run=2)
        
        output_directory = make_dir(f"{os.getcwd()}/results/Resultados_CGA/experimentos/",model_name,a)
        experiment = Experiment(output_dir=output_directory, jobs=jobs, m_workers=os.cpu_count()//2)
        logger.info(f"Running experiment with {len(jobs)} jobs")
        
        experiment.run()

        generate_summary_from_experiment(
            input_dir=output_directory,
            quality_indicators=[FitnessValue(),
                                SelectedVariables()])
        
        file_name = f"{output_directory}/QualityIndicatorSummary.csv"
        generate_latex_tables(filename=file_name,
                                output_dir=output_directory+"/latex/statistical")

# if __name__ == "__main__":
#     cProfile.run('main()', f'{os.getcwd()}/results/Resultados_CGA/experimentos/profile_results.prof')

#     with open(f'{os.getcwd()}/results/Resultados_CGA/experimentos/profile_report.txt', 'w') as f:
#         stats = pstats.Stats(f'{os.getcwd()}/results/Resultados_CGA/experimentos/profile_results.prof', stream=f)
#         stats.strip_dirs()
#         stats.sort_stats('cumulative')
#         stats.print_stats()