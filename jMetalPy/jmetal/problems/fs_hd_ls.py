import pandas as pd
import numpy as np

from Problems import LocalS
from jmetal.algorithm.singleobjective.local_search import LocalSearch
from jmetal.util.termination_criterion import StoppingByEvaluations
from Core import mutation

from sklearn.preprocessing import LabelEncoder,MinMaxScaler


def results(algorithm,experiment,test,clases,params):
    soluciones_ls = algorithm.get_result()
    objectives = soluciones_ls.objectives
    variables = soluciones_ls.variables

    var_squeezed = np.squeeze(variables)
    genes_selected = [gen for gen,var in zip(clases,var_squeezed) if var==1]

    with open(f'Results/Resultados_LS/Resultados_func_agregativa/{experiment}/Resultados_FS_{test}.txt','w') as f:
        f.write(f"Name: {algorithm.get_name()}\n")
        f.write(f"Solucion objectives: {objectives}\n")
        f.write(f"Solucion variables: {variables}\n")
        f.write(f"Solucion variables type: {type(variables)}\n")
        f.write(f"Solucion variables amount: {len(variables)}\n")
        f.write(f"Selected genes: {genes_selected}\n")
        f.write(f"Selected genes amount: {len(genes_selected)}\n")
        f.write(f"Agregative function parameters alfa: {params['alfa']}, beta: {1-params['alfa']}\n")
        f.write(f"Stopping criterion evals : {params['evals']}\n")
        f.write(f"Mutation parameter: {params['mut_p']}\n")
        f.write(f"Clases: {params['encoder'].classes_}\n")

#DATA
df_hd = pd.read_csv('D:/Doctorado/doctorado/Data/HD_filtered.csv')

#PRE-SETS
# encoder = LabelEncoder()
# scaler = MinMaxScaler()
# X = df_hd.drop(columns=['Samples','Grade']).to_numpy()
# X = scaler.fit_transform(X)
# y = encoder.fit_transform(df_hd.Grade.to_numpy())
# clases = list(df_hd.columns[:-2])
df_hd = pd.read_csv('../Data/Leukemia_GSE9476.csv')
print(df_hd.info())
encoder = LabelEncoder()
X = df_hd.drop(columns=['samples','type']).to_numpy()
y = encoder.fit_transform(df_hd.type.to_numpy())
clases = list(df_hd.columns[2:])
print(X.shape)

#PARAMETERS
params = {'evals' : 10,
        'mut_p' :0.01,
        'alfa':0.1,
        'encoder':encoder
        }

#PROBLEM
problem = LocalS.FeatureSelectionLS(X, y, params['alfa'])

#OPERATORS
mut = mutation.BitFlipMutation(params['mut_p'])
criterion = StoppingByEvaluations(max_evaluations=params['evals'])

# # ALGORITHM
# LOCAL SEARCH 
algorithm = LocalSearch(problem,
                        mutation=mut,
                        termination_criterion=criterion
                        )
algorithm.run()

# RESULTS
experiment = 'Experimento3'
test = str(3)
results(algorithm,experiment,test,clases,params)

# algorithm.plot_fitness()