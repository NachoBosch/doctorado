from jmetal.core.solution import BinarySolution
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
from jmetal.util import load

class FeatureSelectionHD():
    def __init__(self, data, alfa, model):
        self.X = data[0]
        self.y = data[1]
        self.alfa = alfa
        self.model = model
        self.number_of_variables = data[0].shape[1]
        self.number_of_objectives = 2  # Ahora guardamos fitness y accuracy
        self.number_of_constraints = 0

    def evaluate(self, solution):
        selected_features = np.flatnonzero(solution.variables)

        if len(selected_features) == 0:
            solution.objectives = [0, 0]  # Sin selección de genes, accuracy 0
            return solution
        
        X_selected = self.X[:, selected_features]
        kf = KFold(n_splits=2, shuffle=True, random_state=42)
        scores = []

        # self.model = load.ann(shape=[len(selected_features)])
        
        for trainI, testI in kf.split(X_selected):
            X_train, X_test = X_selected[trainI], X_selected[testI]
            y_train, y_test = self.y[trainI], self.y[testI]
            self.model.fit(X_train, y_train)#,epochs=50,verbose=0) #se cambió esta linea para trabajar con ANN
            y_pred = self.model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            # acc = accuracy_score(y_test,np.round(np.squeeze(y_pred)))
            scores.append(acc)

        acc_avg = np.mean(scores)  # Accuracy promedio en validación cruzada
        # print(acc_avg)
        num_variables = len(selected_features)
        beta = 1 - self.alfa
        fitness = 1.0 - (num_variables / self.X.shape[1])
        fitness = (self.alfa * fitness) + (beta * acc_avg)

        solution.objectives = [-fitness, acc_avg]  # Guardamos ambos valores
        return solution

    def create_solution(self):
        print("Creando solución")
        new_solution = BinarySolution(
            number_of_variables=self.number_of_variables,
            number_of_objectives=self.number_of_objectives
        )
        new_solution.variables = [True if np.random.rand() > 0.5 else False for _ in range(self.number_of_variables)]
        # new_solution.variables = [np.random.rand() > 0.5 for _ in range(self.number_of_variables)]
        new_solution.objectives = [0, 0]  # Inicializamos fitness y accuracy

        return new_solution

    def get_name(self):
        return "FeatureSelectionHD"
