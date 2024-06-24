from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from snapml import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
import os

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
    return [RandomForestClassifier(n_jobs=os.cpu_count()//2),
            SVC(),
            KNeighborsClassifier(),
            AdaBoostClassifier()]