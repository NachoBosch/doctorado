import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from mealpy import ALO, BinaryVar
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder

from Graphs_results import graph_result
from Algorithms import GeneticAlgorithm, CellularGeneticAlgorithm


#------------------------------------------------------------------------------

#MODELS
def models(name:str='dt'):
  """Selecciona el modelo a entrenar:
                'dt':DecisionTreeClassifier,
                'ab':AdaBoostClassifier,
                'nb':MultinomialNB,
                'rf':RandomForestClassifier,
                'knn':KNeighborsClassifier(),
                'svm':SVC"""
  models_dic = {'dt':DecisionTreeClassifier(),
                'ab':AdaBoostClassifier(),
                'nb':MultinomialNB(),
                'rf':RandomForestClassifier(),
                'knn':KNeighborsClassifier(n_neighbors=9),
                'svm':SVC()}
  return models_dic[name]

#PRE-SETS
df_hd = pd.read_csv('../Data/HD_filtered.csv')
encoder = LabelEncoder()
X = df_hd.drop(columns=['Samples','Grade']).to_numpy()
y = encoder.fit_transform(df_hd.Grade.to_numpy())
clases = list(df_hd.columns[:-2])
print(X.shape)

#Random Forest 
model = models('rf')
xtrain,xtest,ytrain,ytest = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
model.fit(xtrain, ytrain)
print("Baseline Random Forest Score\n")
print(f"Train score: {model.score(xtrain, ytrain)}")
print(f"Test score: {model.score(xtest, ytest)}")

#OBJECTIVE
def fitness_function(solution):
  selected_indices = np.flatnonzero(solution)
  if len(selected_indices)!=0:
    X_new = X[:, selected_indices]

    model = models('rf')
    xtrain,xtest,ytrain,ytest = train_test_split(X_new, y, test_size=0.3, random_state=42, stratify=y)
    model.fit(xtrain,ytrain)
    ypred = model.predict(xtest)

    #Función agregativa: esta función pondera cada termino en la función fitness 
    num_variables = len(selected_indices)
    acc = accuracy_score(ytest, ypred)
    alfa = 0.7
    beta = 1 - alfa
    fitness = 1.0 - (num_variables/X.shape[1]) # Primera parte de la función agregativa
    fitness = (alfa * fitness) + (beta * acc)
    
  else:
    fitness=0

  return fitness

#PROBLEM
problem_dict = {
  "bounds": BinaryVar(n_vars=X.shape[1]),
  "obj_func": fitness_function,
  "minmax": "max",
  "log_file":"rf_result.log"
}

#OPTIMIZER
# optimizer = ALO.OriginalALO(epoch=100, pop_size=50)
# optimizer = GA.SingleGA(epoch=1000, pop_size=150, pc=0.9, pm=0.1, selection = "roulette", crossover = "uniform", mutation = "swap")
#optimizer = GeneticAlgorithm.BaseGA(epoch=10, pop_size=100, pc=0.9, pm=0.01, selection = "tournament", crossover = "one_point", mutation = "flip")
optimizer = CellularGeneticAlgorithm.CellularGA(epoch=10, pop_size=100, pc=0.9, pm=0.01, selection = "tournament", crossover = "one_point", mutation = "flip")
g_best = optimizer.solve(problem_dict)


#RESULTS
selected_indices = np.flatnonzero(g_best.solution)
selected_variables = df_hd.columns[selected_indices]
print(f"Variables seleccionadas: {list(selected_variables)}")
print(f"Cantidad de variables seleccionadas: {len(selected_variables)}")
print(f"Mejor valor de aptitud: {g_best.target.fitness}")

#GRAPHS
prueba = 'CGA'
graph_result(optimizer,prueba)
