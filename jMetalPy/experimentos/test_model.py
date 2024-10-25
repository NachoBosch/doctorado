import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import SpectralBiclustering
from collections import Counter

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.join(current_dir.replace('experimentos', ''))
sys.path.append(module_dir)
from jmetal.problems import classify_models

#DATA
df_hd = pd.read_csv('C:/Doctorado/doctorado/Data/HD_filtered.csv')
df = df_hd.copy()
#PRE-SETS
scaler = MinMaxScaler()
encoder = LabelEncoder()
X = df_hd.drop(columns=['Samples','Grade']).to_numpy()
X = scaler.fit_transform(X)
y = encoder.fit_transform(df_hd.Grade.to_numpy())
clases = list(df_hd.columns[:-2])

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

#PARAMETERS
# params = {'pobl': 100,
#         'off_pobl': 100,
#         'evals' : 10000,
#         'mut_p' :0.1,
#         'cross_p': 0.8,
#         'alfa':0.6,
#         'encoder':encoder
#         }

#PROBLEM
# problem = classify_models.main(X, y, params['alfa'])