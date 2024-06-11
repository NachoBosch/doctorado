from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import xgboost as xgb

from sklearn import metrics as ms
import numpy as np 

def main(X,y,alfa):

    def models_to_train():
        return {'dt':DecisionTreeClassifier(),
                'ab':AdaBoostClassifier(),
                'rf':RandomForestClassifier(),
                'svm':SVC(),
                'knn':KNeighborsClassifier(),
                'xgb':xgb.XGBClassifier(eta=0.01)}
        
    def train_test(X,y):
        Xtrain,Xtest,ytrain,ytest = train_test_split(X, y,
                                                    test_size=0.25,
                                                    stratify=y)
        models = models_to_train()
        for model in models.keys():
            fitter = models[model]
            fitter.fit(Xtrain, ytrain)
            y_pred = fitter.predict(Xtest)
            acc = ms.accuracy_score(ytest, y_pred)
            print(f"Accuracy of {model}: {acc:.3f}")

    def evaluate(X,y,alfa):
        random_variables = np.random.randint(0, 2, X.shape[1])
        X_selected = X[:, random_variables == 1]
        Xtrain,Xtest,ytrain,ytest = train_test_split(X_selected, y,
                                                    test_size=0.25,
                                                    stratify=y)
        print(Xtrain.shape)
        models = models_to_train()
        for model in models.keys():
            fitter = models[model]
            fitter.fit(Xtrain, ytrain)
            y_pred = fitter.predict(Xtest)
            acc = ms.accuracy_score(ytest, y_pred)
            num_variables = len(X_selected)
            beta = 1 - alfa
            fitness = 1.0 - (num_variables/X.shape[1])
            fitness = (alfa * fitness) + (beta * acc)
            print(f"Fitness of {model}: {fitness:.3f} | Variables: {num_variables}")

    train_test(X,y)
    evaluate(X,y,alfa)

if __name__ == '__main__':
    main(X,y,alfa)
