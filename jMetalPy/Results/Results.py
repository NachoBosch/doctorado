
import numpy as np


def results(algorithm,experiment,test,clases,params):
    soluciones_ls = algorithm.get_result()
    objectives = soluciones_ls.objectives
    variables = soluciones_ls.variables

    var_squeezed = np.squeeze(variables)
    genes_selected = [gen for gen,var in zip(clases,var_squeezed) if var==1]

    with open(f'Results/Resultados_CGA/Resultados_func_agregativa/{experiment}/Resultados_FS_{test}.txt','w') as f:
        f.write(f"Name: {algorithm.get_name()}\n")
        f.write(f"Solucion objectives: {objectives}\n")
        f.write(f"Solucion variables: {variables}\n")
        f.write(f"Solucion variables type: {type(variables)}\n")
        f.write(f"Solucion variables amount: {len(variables)}\n")
        f.write(f"Selected genes: {genes_selected}\n")
        f.write(f"Selected genes amount: {len(genes_selected)}\n")
        f.write(f"Agregative function parameters alfa: {params['alfa']}, beta: {1-params['alfa']}\n")
        f.write(f"Stopping criterion evals : {params['evals']}\n")
        f.write(f"Crossover parameter: {params['cross_p']}\n")
        f.write(f"Mutation parameter: {params['mut_p']}\n")
        f.write(f"Poblation size: {params['pobl']} | Offspring size: {params['off_pobl']}\n")
        f.write(f"Clases: {params['encoder'].classes_}\n")