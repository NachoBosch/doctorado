from jmetal.core.solution import BinarySolution
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


class FeatureSelectionHD():
    def __init__(self,data,alfa,model):
        self.X = data[0]
        self.y = data[1]
        self.alfa = alfa
        self.model = model
        self.number_of_variables = data[0].shape[1]
        self.number_of_objectives = 1
        self.number_of_constraints = 0

    def evaluate(self, solution):
        selected_features = np.flatnonzero(solution.variables)
        # print(f"Selected Features: {len(selected_features)}")
        if len(selected_features) == 0:
            solution.objectives[0] = 0
            return solution
        X_selected = self.X[:, selected_features]
        kf = KFold(n_splits=2, shuffle=True, random_state=42)
        scores = []
        for trainI, testI in kf.split(X_selected):
            X_train, X_test = X_selected[trainI], X_selected[testI]
            y_train, y_test = self.y[trainI], self.y[testI]
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            scores.append(acc)

        acc_avg = np.mean(scores)

        num_variables = len(selected_features)
        beta = 1 - self.alfa
        gens = 1.0 - (num_variables/self.X.shape[1])
        fitness = (self.alfa * gens) + (beta * acc_avg)
        solution.objectives[0] = -fitness
        return solution

    def create_solution(self):
        new_solution = BinarySolution(
            number_of_variables = self.number_of_variables,
            number_of_objectives = self.number_of_objectives)
        
        new_solution.variables = [True if np.random.rand() > 0.5 else False for _ in range(self.number_of_variables)]#probar con sesgo
        # print(f"New solution variables: {len(new_solution.variables)}")
        new_solution.objectives = [0]

        return new_solution

    def get_name(self):
        return "FeatureSelectionHD"
