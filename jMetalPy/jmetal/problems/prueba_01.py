import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn import metrics as ms

# Cargamos dataset de prueba
data = pd.read_csv('../Data/MSFT.csv')
print(f"Observamos algunos registros:\n {data.head(2)}")

print(f"Observamos info del dataset:\n {data.info()}")

plt.figure()
data['Close'].plot(figsize=(10,7))
plt.show()

print(f"Correlación de las variables: {data.corr()}")

X = data[['Open','High']].to_numpy()
y = data['Close'].to_numpy()
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=8)

# Leemos los parámetros
with open('GeneticAlgorithmOptimizer.txt','r') as f:
  file = f.read()

print(file)

# Utilizamos los parámetros que devuelve el proceso de optimización
# svr = SVR(kernel='linear',C=,degree=1,epsilon=solution.variables[1])
# svr.fit(Xtrain,ytrain)
# ypred = svr.predict(Xtest)
# mae = ms.mean_absolute_error(ytest,ypred)
# print(f"MAE : {mae}")

# plt.figure()
# plt.scatter(Xtest[:,0],ytest,c='orange')
# plt.scatter(Xtest[:,0],ypred,c='green')
# plt.show()