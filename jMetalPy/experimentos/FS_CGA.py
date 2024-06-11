import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.join(current_dir.replace('experimentos', ''))
sys.path.append(module_dir)

from jmetal.lab.experiment import Experiment, Job, generate_summary_from_experiment,generate_latex_tables
from jmetal.core.quality_indicator import *
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util import load
from jmetal.problems import FeatureSelectionHutington as fsh
from jmetal.algorithms.cga import CellularGeneticAlgorithm
from jmetal.core import crossover, mutation, selection
from jmetal.util.observer import PrintObjectivesObserver
from jmetal.util.neighborhood import L5
from jmetal.util.update_policy import LineSweep


def configure_experiment(problems: dict,
                         cxp: list,
                         cmp: float, 
                         n_run: int):
    jobs = []
    for cx in cxp:
        for run in range(n_run):
            for problem_tag, problem in problems.items():
                jobs.append(
                    Job(
                    algorithm=CellularGeneticAlgorithm(
                            problem = problem,
                            pop_size = 25,
                            mutation = mutation.BitFlipMutation(cmp),
                            crossover = crossover.SPXCrossover(cx),
                            selection = selection.BinaryTournamentSelection(),
                            termination_criterion=StoppingByEvaluations(50),
                            neighborhood=L5(rows=5,columns=5),
                            cell_update_policy=LineSweep()
                        ),
                    algorithm_tag="CGA",
                    problem_tag=problem_tag,
                    run=run,
                )
            )

    return jobs


if __name__ == "__main__":
    
    data = load.huntington()
    alfa = [0.9]
    models = load.models()
    cxp = [0.9]
    cmp = 0.01
    for model in models[:1]:
        for a in alfa:
            print(f"Model: {model} | Alfa {a}")
            jobs = configure_experiment(problems={"FS_CGA": fsh.FeatureSelectionHD(data,a,model)}, 
                                        cxp = cxp,
                                        cmp = cmp,
                                        n_run=2)
            output_directory = f"{os.getcwd()}/results/Resultados_CGA/experimentos"
            experiment = Experiment(output_dir=output_directory, jobs=jobs)

            experiment.run()

            generate_summary_from_experiment(
                input_dir=output_directory,
                quality_indicators=[FitnessValue()])
            
            file_name = f"{output_directory}/QualityIndicatorSummary.csv"
            generate_latex_tables(filename=file_name,
                                  output_dir=output_directory+"/latex/statistical")

