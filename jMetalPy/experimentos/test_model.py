import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import SpectralBiclustering
from collections import Counter
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.join(current_dir.replace('experimentos', ''))
sys.path.append(module_dir)
from jmetal.problems import classify_models

def stats(X):
    mean = np.mean(X).round(2)
    std = np.std(X).round(2)
    max = np.max(X).round(2)
    min = np.min(X).round(2)
    return {'mean':mean,
            'std':std,
            'max':max,
            'min':min
            }
#DATA
df_hd = pd.read_csv('C:/Doctorado/doctorado/Data/HD_filtered.csv')
df = df_hd.copy()

#PRE-SETS
scaler = MinMaxScaler()
encoder = LabelEncoder()
X = df.drop(columns=['Samples','Grade'])

print(len(X.columns))
X_scaled = scaler.fit_transform(X)

# var_thresh = VarianceThreshold(threshold=0.05)
# X_var_filtered = var_thresh.fit_transform(X_scaled)
# selected_var_indices = var_thresh.get_support(indices=True)
# selected_var_names = X.columns[selected_var_indices]
# print(len(selected_var_names))

# y = encoder.fit_transform(df.Grade.to_numpy())
y = df.Grade
# mi_scores = mutual_info_classif(X_var_filtered, y, random_state=42)

# feature_importance = pd.DataFrame({
#     'Feature': selected_var_names,
#     'Mutual_Information': mi_scores
# }).sort_values(by='Mutual_Information', ascending=False)

# top_features = feature_importance.head(50)['Feature'].tolist()
# X_top = X[top_features]
# print(len(X_top.columns))
# print(f"\nSelected Top Features: {list(X_top.columns)}")
df['Grade'] = df['Grade'].replace({'HD_0': 'HD','HD_1': 'HD','HD_2': 'HD','HD_3': 'HD', 'HD_4': 'HD'})
y = encoder.fit_transform(df.Grade.to_numpy())
print(y)
clases = list(df.columns[:-2])

'''
df.drop(columns="Samples",inplace=True)

df_transp = df.T

columnas = df_transp.iloc[-1]
df_transp.columns = columnas
df_transp.drop(df_transp.index[-1],inplace=True)

# fig = px.imshow(df_transp, color_continuous_scale='RdBu', aspect="auto")
# fig.show()

df_num = df_transp.copy()
df_num = df_num.apply(pd.to_numeric, errors='coerce')

# print(df_num.dtypes)

model = SpectralBiclustering(n_clusters=(5, 5), random_state=0)
model.fit(df_num)

fit_data = df_num.iloc[np.argsort(model.row_labels_)]
fit_data = fit_data.iloc[:, np.argsort(model.column_labels_)]

# Visualizar los biclusters como un mapa de calor
# plt.figure(figsize=(10, 8))
# sns.heatmap(fit_data, cmap="viridis")
# plt.title("Heatmap de Biclustering")
# plt.show()

# # Imprimir las etiquetas de las filas y columnas
# print("Etiquetas de filas:")
# print(model.row_labels_)
# print("Etiquetas de columnas:")
# print(model.column_labels_)

# sns.clustermap(df_num, row_cluster=True, col_cluster=True, row_linkage=model.row_labels_, col_linkage=model.column_labels_)
# plt.show()

# fig = px.imshow(df_transp, color_continuous_scale='RdBu', aspect="auto")
# fig.show()

genes = df_num.index
conditions = df_num.columns

# Reordenar los datos en función del clustering realizado
row_clusters = model.row_labels_
column_clusters = model.column_labels_

# Agrupar genes y condiciones por bicluster
for bicluster in np.unique(row_clusters):
    print(f"Bicluster {bicluster}:")
    
    # Genes en este bicluster
    genes_in_bicluster = genes[row_clusters == bicluster]
    print(f"Genes: {len(list(genes_in_bicluster))}| {list(genes_in_bicluster)}")
    
    # Condiciones en este bicluster
    conditions_in_bicluster = conditions[column_clusters == bicluster]
    print(f"Condiciones: {list(conditions_in_bicluster)}")
    print('-' * 50)

# Diccionario para almacenar los genes de cada bicluster
genes_in_biclusters = {}

# Agrupar genes por bicluster
for bicluster in np.unique(row_clusters):
    genes_in_bicluster = genes[row_clusters == bicluster]
    genes_in_biclusters[bicluster] = list(genes_in_bicluster)

# Contar cuántas veces aparece cada gen en todos los biclusters
gene_counter = Counter()

for gene_list in genes_in_biclusters.values():
#     print(gene_list)
    gene_counter.update(gene_list)

# Filtrar genes que se repiten en al menos 2 biclusters
repeated_genes = [gene for gene, count in gene_counter.items() if count >= 2]

# Imprimir los genes que se repiten en al menos 2 biclusters
print("Genes que se repiten en al menos 2 biclusters:")
print(repeated_genes)

'''

#PARAMETERS
params = {'pobl': 100,
        'off_pobl': 100,
        'evals' : 10000,
        'mut_p' :0.1,
        'cross_p': 0.8,
        'alfa':0.9,
        'encoder':encoder
        }

# # PROBLEM
problem = classify_models.main(X.to_numpy(), y, params['alfa'])


