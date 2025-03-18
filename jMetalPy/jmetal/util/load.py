from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.ensemble import BaggingClassifier
from snapml import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
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
    df_hd['Grade'] = df_hd['Grade'].replace({'HD_0': 'HD','HD_1': 'HD','HD_2': 'HD','HD_3': 'HD', 'HD_4': 'HD'})
    y = encoder.fit_transform(df_hd.Grade.to_numpy())
    clases = list(df_hd.columns[:-2])
    return X,y,clases,encoder

def huntington_bic():
    path = os.getcwd().replace('jMetalPy','')
    df_hd = pd.read_csv(path+'Data/HD_filtered.csv')
    df_hd.drop(columns="Samples",inplace=True)
    df_transp = df_hd.T
    columnas = df_transp.iloc[-1]
    df_transp.columns = columnas
    df_transp.drop(df_transp.index[-1],inplace=True)
    df_num = df_transp.apply(pd.to_numeric, errors='coerce')
    scaler = MinMaxScaler()
    df_num.loc[:, :] = scaler.fit_transform(df_num)
    return df_num

def models():
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)

        models = [RandomForestClassifier(n_jobs=os.cpu_count()//2),
                BaggingClassifier(SVC(), max_samples=1.0/2, n_estimators=2),
                KNeighborsClassifier(n_neighbors=3, weights='distance'),
                AdaBoostClassifier(n_estimators=5),
                DecisionTreeClassifier(max_depth=20)]
                #MLPClassifier(random_state=1, max_iter=300,alpha=0.001)
                
        models_names = ['RF','SVM','KNN','AB','DT']#,'MLP']
        
        return models_names, models