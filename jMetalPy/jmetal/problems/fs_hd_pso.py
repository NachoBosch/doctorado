import pandas as pd
import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
module_dir = os.path.join(current_dir.replace('jmetal\problems', ''))
print(module_dir)
sys.path.append(module_dir)

from jmetal.algorithms.PSO import PSOAlgorithm
from jmetal.problems import GA 
from results import Results
from util.termination_criterion import StoppingByEvaluations
from util.observer import PrintObjectivesObserver

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split,KFold


#DATA
df_hd = pd.read_csv('../Data/HD_filtered.csv')

#PRE-SETS
encoder = LabelEncoder()
X = df_hd.drop(columns=['Samples','Grade']).to_numpy()
y = encoder.fit_transform(df_hd.Grade.to_numpy())
clases = list(df_hd.columns[:-2])

#PARAMETERS
params = {'pobl': 5,
        'evals' : 1000,
        'inertia_weight' :0.7,
        'cognitive_coefficient': 1.4,
        'social_coefficient':1.4,
        'alfa':0.9,
        'encoder':encoder
        }

#BASELINE
# Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.3,random_state=42)
kf = KFold(n_splits=4, shuffle=True, random_state=42)
score = []
model = SVC()
for trainIndex, testIndex in kf.split(X):
    X_train, X_test = X[trainIndex], X[testIndex]
    y_train, y_test = y[trainIndex], y[testIndex]
    model.fit(X_train, y_train)
    sc= model.score(X_test, y_test)
    score.append(sc)

print(f"Score avg: {np.mean(score)}")

#PROBLEM
problem = GA.FeatureSelectionGA(X, y, params['alfa'])

#OPERATORS
mut = mutation.BitFlipMutation(params['mut_p'])
cross = crossover.SPXCrossover(params['cross_p'])
selection = selection.BinaryTournamentSelection()

# # ALGORITHM
algorithm = PSOAlgorithm(
    problem=problem,
    population_size=params['pobl'],
    inertia_weight=params['inertia_weight'],
    cognitive_coefficient=params['cognitive_coefficient'],
    social_coefficient=params['social_coefficient'],
    reinicio=params['reinicio'],
    termination_criterion=StoppingByEvaluations(params['evals'])
)
algorithm.observable.register(observer=PrintObjectivesObserver(10))
algorithm.run()

# RESULTS
test_name = 'First_test'
Results.results(algorithm,test_name,clases,params)
# algorithm.plot_fitness()
