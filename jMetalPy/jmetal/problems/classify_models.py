from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
# from thundersvm import SVC as tSVC
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import xgboost as xgb
from snapml import RandomForestClassifier as snapRF
# from snapml import SupportVectorMachine as snapsvm
from sklearn.svm import NuSVC
from snapml import DecisionTreeClassifier as snapdt
import os
from sklearn import metrics as ms
import numpy as np 
import time

def main(X,y,alfa):

    def models_to_train():
        return {'dt':DecisionTreeClassifier(),
                'snapdt':snapdt(max_depth=1),
                'ab25':AdaBoostClassifier(n_estimators=10),
                'ab50':AdaBoostClassifier(n_estimators=50),
                'snapab':AdaBoostClassifier(base_estimator=snapdt(max_depth=1)),
                'rf':RandomForestClassifier(max_depth=12,n_jobs=os.cpu_count()//2),
                'snaprf':snapRF(),
                'svmNuSVC':NuSVC(kernel='rbf',nu=0.01),
                'svm':SVC(cache_size=100),
                'svmBag1':BaggingClassifier(SVC(),max_samples=1.0/2,n_estimators=2),
                'knn':KNeighborsClassifier(),
                'xgb':xgb.XGBClassifier(eta=0.01)}
        
    def train_test(X, y):
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, stratify=y)
        models = models_to_train()
        for model_name, model in models.items():
            start_time = time.time()
            model.fit(Xtrain, ytrain)
            end_time = time.time()
            y_pred = model.predict(Xtest)
            acc = ms.accuracy_score(ytest, y_pred)
            print(f"Accuracy of {model_name}: {acc:.3f} | Training time: {end_time - start_time:.2f} seconds")

    def evaluate(X, y, alfa):
        random_variables = np.random.randint(0, 2, X.shape[1])
        X_selected = X[:, random_variables == 1]
        Xtrain, Xtest, ytrain, ytest = train_test_split(X_selected, y, test_size=0.25, stratify=y)
        print(Xtrain.shape)
        models = models_to_train()
        for model_name, model in models.items():
            start_time = time.time()
            model.fit(Xtrain, ytrain)
            end_time = time.time()
            y_pred = model.predict(Xtest)
            acc = ms.accuracy_score(ytest, y_pred)
            num_variables = X_selected.shape[1]
            beta = 1 - alfa
            fitness = 1.0 - (num_variables / X.shape[1])
            fitness = (alfa * fitness) + (beta * acc)
            print(f"Fitness of {model_name}: {fitness:.3f} | Variables: {num_variables} | Training time: {end_time - start_time:.2f} seconds")


    train_test(X,y)
    evaluate(X,y,alfa)

if __name__ == '__main__':
    main(X,y,alfa)
