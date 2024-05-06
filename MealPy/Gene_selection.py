import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime
sns.set()

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import mealpy
from mealpy import FloatVar, ALO, BinaryVar, GA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder

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

## Pre-filtering Select K-best
kbest = SelectKBest(score_func=f_classif, k=100)
X_select = kbest.fit_transform(X, y)
print("Columnas seleccionadas KBest:", len(kbest.get_support(indices=True)))
selected_features = [clases[i] for i in kbest.get_support(indices=True)]
print(f"Features seleccionadas KBest: {selected_features}")

#Random Forest 
model = models('rf')
xtrain,xtest,ytrain,ytest = train_test_split(X_select,y,test_size=0.3,random_state=42,stratify=y)
# xtrain,xtest,ytrain,ytest = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
model.fit(xtrain, ytrain)
print("Baseline Random Forest Score\n")
print(f"Train score: {model.score(xtrain, ytrain)}")
print(f"Test score: {model.score(xtest, ytest)}")

#OBJECTIVE
def fitness_function(solution):
  selected_indices = np.flatnonzero(solution)
  if len(selected_indices)!=0:
    # X_new = X[:, selected_indices]
    X_new = X_select[:, selected_indices]

    model = models('rf')
    xtrain,xtest,ytrain,ytest = train_test_split(X_new, y, test_size=0.3, random_state=42, stratify=y)
    model.fit(xtrain,ytrain)
    ypred = model.predict(xtest)

    #Función agregativa: esta función pondera cada termino en la función fitness 
    num_variables = len(selected_indices)
    acc = accuracy_score(ytest, ypred)
    alfa = 0.3
    beta = 1 - alfa
    fitness = 1.0 - (num_variables/X_select.shape[1]) # Primera parte de la función agregativa
    fitness = (alfa * fitness) + (beta * acc)
    
    #Función agregativa sin ponderar
    #num_variables = len(selected_indices)
    #acc = accuracy_score(ytest, ypred)
    #penalizacion = num_variables/X_select.shape[1]
    #fitness = acc - penalizacion
  else:
    fitness=0

  return fitness

#PROBLEM
problem_dict = {
  "bounds": BinaryVar(n_vars=X_select.shape[1]),
  "obj_func": fitness_function,
  "minmax": "max",
  "log_file":"rf_result.log"
}

#OPTIMIZER
# optimizer = ALO.OriginalALO(epoch=100, pop_size=50)
# optimizer = GA.SingleGA(epoch=1000, pop_size=150, pc=0.9, pm=0.1, selection = "roulette", crossover = "uniform", mutation = "swap")
optimizer = GA.BaseGA(epoch=10, pop_size=150, pc=0.9, pm=0.01, selection = "tournament", crossover = "one_point", mutation = "flip")
g_best = optimizer.solve(problem_dict)


#RESULTS
selected_indices = np.flatnonzero(g_best.solution)
selected_variables = df_hd.columns[selected_indices]
print(f"Variables seleccionadas: {list(selected_variables)}")
print(f"Cantidad de variables seleccionadas: {len(selected_variables)}")
print(f"Mejor valor de aptitud: {g_best.target.fitness}")