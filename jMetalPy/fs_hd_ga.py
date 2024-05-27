import pandas as pd

from Algorithms.GA import GeneticAlgorithm
from Problems import GA
from Results import Results
from Core import crossover, mutation, selection

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


#DATA
df_hd = pd.read_csv('../Data/HD_filtered.csv')

#PRE-SETS
encoder = LabelEncoder()
X = df_hd.drop(columns=['Samples','Grade']).to_numpy()
y = encoder.fit_transform(df_hd.Grade.to_numpy())
clases = list(df_hd.columns[:-2])

#PARAMETERS
params = {'pobl': 150,
        'off_pobl': 150,
        'evals' : 15000,
        'mut_p' :0.1,
        'cross_p': 0.9,
        'alfa':0.9,
        'encoder':encoder
        }

#BASELINE
Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.3,random_state=42)
model = SVC()
model.fit(Xtrain, ytrain)
print(f"Train: {model.score(Xtrain,ytrain)}")
print(f"Test: {model.score(Xtest,ytest)}")

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
experiment = 'Experimento6'
test = str(2)
Results.results(algorithm,experiment,test,clases,params)

algorithm.plot_fitness()
