import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.join(current_dir.replace('experimentos', ''))
sys.path.append(module_dir)
from jmetal.problems import classify_models

#DATA
df_hd = pd.read_csv('C:/Doctorado/doctorado/Data/HD_filtered.csv')
df = df_hd.copy()
#PRE-SETS
scaler = MinMaxScaler()
encoder = LabelEncoder()
X = df_hd.drop(columns=['Samples','Grade']).to_numpy()
X = scaler.fit_transform(X)
y = encoder.fit_transform(df_hd.Grade.to_numpy())
clases = list(df_hd.columns[:-2])

df.drop(columns="Samples",inplace=True)
print(df.T)


#PARAMETERS
params = {'pobl': 100,
        'off_pobl': 100,
        'evals' : 10000,
        'mut_p' :0.1,
        'cross_p': 0.8,
        'alfa':0.6,
        'encoder':encoder
        }

#PROBLEM
# problem = classify_models.main(X, y, params['alfa'])