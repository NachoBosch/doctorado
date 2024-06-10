
import os
import sys

# Obtener la ruta del directorio actual
current_dir = os.path.dirname(os.path.abspath(__file__))

# Reemplazar 'experimentos' con 'jmetal' en la ruta
module_dir = os.path.join(current_dir.replace('experimentos', ''))

print(module_dir)
# Añadir el directorio jmetal al sys.path
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
                            pop_size = 100,
                            mutation = mutation.BitFlipMutation(cmp),
                            crossover = crossover.SPXCrossover(cx),
                            selection = selection.BinaryTournamentSelection(),
                            termination_criterion=StoppingByEvaluations(1000),
                            neighborhood=L5(rows=10,columns=10),
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
    alfa = [0.7, 0.9]
    models = load.models()
    cxp = [0.8,0.9]
    cmp = 0.01
    for model in models:
        for a in alfa:
            print(f"Model: {model} | Alfa {a}")
            jobs = configure_experiment(problems={"FS_CGA": fsh.FeatureSelectionHD(data,a,model)}, 
                                        cxp = cxp,
                                        cmp = cmp,
                                        n_run=2)
            output_directory = "data"
            experiment = Experiment(output_dir=output_directory, jobs=jobs)
            # experiment.observable.register(observer=PrintObjectivesObserver(100))
            experiment.run()

            file_name = f"../results/Resultados_CGA/QualityIndicatorSummary{model.get_name()}_alfa_{str(a)}.csv"
            generate_latex_tables(filename=file_name)

