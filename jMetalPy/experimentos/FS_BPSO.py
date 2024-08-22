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
from jmetal.algorithms.BPSO import BinaryPSOAlgorithm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_dir(path,model_name,alfa):
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
                algorithm = BinaryPSOAlgorithm(
                        problem = problem,
                        swarm_size=100,
                        inertia_weight=0.7,
                        cognitive_coefficient=1.4,
                        social_coefficient=1.4,
                        termination_criterion=StoppingByEvaluations(10000)
                    ),
                algorithm_tag="BPSO",
                problem_tag=problem_tag,
                run=run)
            )
    return jobs

data = load.huntington()
alfa = 0.9
models_names, models = load.models()
for model_name, model in zip(models_names,models):
    jobs = configure_experiment(problems={"FS_BPSO": fsh.FeatureSelectionHD(data,alfa,model)},
                                n_run=20)
    
    output_directory = make_dir(f"{os.getcwd()}/results/Resultados_BPSO/experimentos/",model_name,alfa)
    experiment = Experiment(output_dir=output_directory, jobs=jobs, m_workers=os.cpu_count())
    logger.info(f"Running experiment with {len(jobs)} jobs")
    
    experiment.run()

    generate_summary_from_experiment(
        input_dir=output_directory,
        quality_indicators=[FitnessValue(),
                            SelectedVariables()])
    
    file_name = f"{output_directory}/QualityIndicatorSummary.csv"
    generate_latex_tables(filename=file_name,
                            output_dir=output_directory+"/latex/statistical")
