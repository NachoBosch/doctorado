import pandas as pd
import numpy as np

# from Algorithms.GA import GeneticAlgorithm
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.join(current_dir.replace('experimentos', ''))
sys.path.append(module_dir)
from Algorithms.GA_LO import GeneticAlgorithm
from Results import Results
from Core import crossover, mutation, selection

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from Algorithms import NeuralNet
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
params = {'pobl': 100,
        'off_pobl': 100,
        'evals' : 1000,
        'mut_p' :0.01,
        'cross_p': 0.9,
        'alfa':0.9,
        'encoder':encoder
        }

#BASELINE
# Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.3,random_state=42)
kf = KFold(n_splits=4, shuffle=True, random_state=42)
score = []
model = MLPClassifier(max_iter=500)
for trainIndex, testIndex in kf.split(X):
    X_train, X_test = X[trainIndex], X[testIndex]
    y_train, y_test = y[trainIndex], y[testIndex]
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    # model = NeuralNet.train_nn(X_train, y_train)
    # y_pred = np.argmax(np.squeeze(model.predict(X_test,verbose=0)),axis=1)
    sc= accuracy_score(y_test,y_pred)
    score.append(sc)

print(f"Score avg: {np.mean(score)}")

#PROBLEM
problem = GA.FeatureSelectionGA(X, y, params['alfa'])

#OPERATORS
mut = mutation.BitFlipMutation(params['mut_p'])
cross = crossover.SPXCrossover(params['cross_p'])
selection = selection.BinaryTournamentSelection()

# # ALGORITHM
algorithm = GeneticAlgorithm(
    problem=problem,
    population_size=params['pobl'],
    offspring_population_size=params['off_pobl'],
    mutation=mut,
    crossover=cross,
    selection=selection,
    max_evaluations=params['evals']
)

algorithm.run()

# RESULTS
test = 'MLP'
Results.results(algorithm,test,clases,params)

algorithm.plot_fitness()
algorithm.plot_min_variables()
algorithm.save_csv(f'results/Resultados_GALO/Resultados_nuevos/{test}.csv')
