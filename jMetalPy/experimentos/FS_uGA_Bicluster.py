import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.join(current_dir.replace('experimentos', ''))
sys.path.append(module_dir)

from jmetal.lab.experiment import Experiment, Job, generate_summary_from_experiment,generate_latex_tables
from jmetal.core.quality_indicator import *
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util import load
from jmetal.problems import Bic
from jmetal.algorithms.uGA import MicroGeneticAlgorithm
from jmetal.core import crossover, mutation, selection
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_dir(path,model_name):
    path = path+model_name
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
                algorithm=MicroGeneticAlgorithm(
                        problem = problem,
                        population_size= 10,
                        mutation = mutation.BitFlipMutation(0.01),
                        crossover = crossover.SPXCrossover(0.9),
                        selection = selection.BinaryTournamentSelection(),
                        reinicio = 50,
                        termination_criterion=StoppingByEvaluations(10000)
                    ),
                algorithm_tag="uGA",
                problem_tag=problem_tag,
                run=run)
            )
    return jobs

data = load.huntington_bic()
model_name = "BIC"
jobs = configure_experiment(problems={"BIC_uGA": Bic.BiclusteringProblem(data)},
                            n_run=10)

output_directory = make_dir(f"{os.getcwd()}/results/Resultados_uGA/experimentos/",model_name)
experiment = Experiment(output_dir=output_directory, jobs=jobs, m_workers=os.cpu_count())
logger.info(f"Running experiment with {len(jobs)} jobs")

experiment.run()

generate_summary_from_experiment(
    input_dir=output_directory,
    quality_indicators=[FitnessValue(),
                        SelectedBicluster()])

file_name = f"{output_directory}/QualityIndicatorSummary.csv"
generate_latex_tables(filename=file_name,
                        output_dir=output_directory+"/latex/statistical")
