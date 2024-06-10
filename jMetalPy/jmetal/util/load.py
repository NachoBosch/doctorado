from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

def huntington():
    df_hd = pd.read_csv('C:/Doctorado/doctorado/Data/HD_filtered.csv')
    encoder = LabelEncoder()
    scaler = MinMaxScaler()
    X = df_hd.drop(columns=['Samples','Grade']).to_numpy()
    X = scaler.fit_transform(X)
    y = encoder.fit_transform(df_hd.Grade.to_numpy())
    clases = list(df_hd.columns[:-2])
    return X,y,clases,encoder

def models():
    return [RandomForestClassifier(),
            SVC(),
            KNeighborsClassifier(),
            AdaBoostClassifier()]