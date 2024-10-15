from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.ensemble import BaggingClassifier
from snapml import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

def huntington():
    path = os.getcwd().replace('jMetalPy','')
    df_hd = pd.read_csv(path+'Data/HD_filtered.csv')
    encoder = LabelEncoder()
    scaler = MinMaxScaler()
    X = df_hd.drop(columns=['Samples','Grade']).to_numpy()
    X = scaler.fit_transform(X)
    y = encoder.fit_transform(df_hd.Grade.to_numpy())
    clases = list(df_hd.columns[:-2])
    return X,y,clases,encoder

def models():
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)

        models = [RandomForestClassifier(n_jobs=os.cpu_count()//2),
                BaggingClassifier(SVC(), max_samples=1.0/2, n_estimators=2),
                KNeighborsClassifier(n_neighbors=3, weights='distance'),
                AdaBoostClassifier(n_estimators=5),
                DecisionTreeClassifier(max_depth=20)
                ]
        models_names = ['RF','SVM','KNN','AB','DT']
        
        return models_names, models