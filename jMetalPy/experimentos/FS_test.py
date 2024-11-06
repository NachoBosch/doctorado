import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.join(current_dir.replace('experimentos', ''))
sys.path.append(module_dir)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from itertools import chain
'''
results = {}

alg_ls = ['BACO','BPSO']

for alg in alg_ls:
    dir_name = f'results/Resultados_toko/Resultados_{alg}/experimentos'
    models_names = os.listdir(dir_name)
    print(f"ALgorithm : {alg}")
    results[alg] = {}
    for model in models_names:
        files = os.path.join(dir_name,model)+f"/alfa_0.9/{alg}/FS_{alg}"
        print(model)
        results[alg][model] = []
        for file in os.listdir(files):
            if file[:3]=='VAR':
                df = pd.read_csv(os.path.join(files,file),sep='\t')
                ls_var = df.columns.to_list()[0].split(' ')
                ls_var.pop(-1)
                # print(len(ls_var))
                ls_genes = [i == 'True' for i in ls_var]
                idx_genes = [i for i, x in enumerate(ls_genes) if x]
                # print(ls_genes)
                print(idx_genes)
                results[alg][model].append(idx_genes)

result = pd.DataFrame()

for alg, alg_results in results.items():
    for model, model_results in alg_results.items():
        for idx_list in model_results:
            result = result.append({
                'Algorithm': alg,
                'Model': model,
                'GenesIdx': idx_list
            }, ignore_index=True)

print(result)

def analyze_algorithm_genes(df, algorithm, top_n=20):
    """
    Analiza y visualiza la frecuencia de índices de genes para un algoritmo específico.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame con columnas 'Algorithm', 'Model' y 'GenesIdx'
    algorithm : str
        Nombre del algoritmo a analizar ('BACO' o 'BPSO')
    top_n : int
        Número de genes más frecuentes a mostrar
        
    Returns:
    --------
    tuple
        (estadísticas, figura)
    """
    # Filtrar datos para el algoritmo específico
    df_alg = df[df['Algorithm'] == algorithm].copy()
    
    # Crear figura
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Obtener todas las listas de índices para este algoritmo
    all_indices = list(chain.from_iterable(df_alg['GenesIdx']))
    freq_dict = Counter(all_indices)
    
    # Ordenar por frecuencia
    sorted_freq = dict(sorted(freq_dict.items(), key=lambda x: x[1], reverse=True))
    
    # Top N genes más frecuentes
    top_indices = list(sorted_freq.keys())[:top_n]
    top_freqs = list(sorted_freq.values())[:top_n]
    
    # 1. Gráfico de barras de frecuencia general
    sns.barplot(x=top_indices, y=top_freqs, ax=ax1, color='skyblue')
    ax1.set_title(f'Frecuencia de Índices de Genes para {algorithm} (Top {top_n})')
    ax1.set_xlabel('Índice del Gen')
    ax1.set_ylabel('Frecuencia')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Análisis por modelo
    freq_by_model = {}
    for model in df_alg['Model'].unique():
        indices_model = list(chain.from_iterable(df_alg[df_alg['Model'] == model]['GenesIdx']))
        freq_by_model[model] = Counter(indices_model)
    
    # Crear matriz para heatmap
    heatmap_data = []
    models = list(freq_by_model.keys())
    for model in models:
        row = [freq_by_model[model].get(idx, 0) for idx in top_indices]
        heatmap_data.append(row)
    
    # Graficar heatmap por modelo
    sns.heatmap(heatmap_data, 
                xticklabels=top_indices,
                yticklabels=models,
                ax=ax2,
                cmap='YlOrRd',
                annot=True,
                fmt='d')
    ax2.set_title(f'Frecuencia de Índices por Modelo - {algorithm}')
    ax2.set_xlabel('Índice del Gen')
    ax2.set_ylabel('Modelo')
    
    plt.tight_layout()
    
    # Calcular estadísticas
    stats = {
        'total_genes_selected': len(all_indices),
        'unique_genes': len(freq_dict),
        'most_common_genes': dict(list(sorted_freq.items())[:5]),
        'avg_genes_per_run': len(all_indices) / len(df_alg),
        'genes_by_model': freq_by_model,
        'total_runs': len(df_alg)
    }
    
    return stats, fig

def print_algorithm_stats(stats, algorithm):
    """
    Imprime las estadísticas de manera formateada para un algoritmo.
    """
    print(f"\nEstadísticas para {algorithm}:")
    print("="*50)
    print(f"Total de ejecuciones: {stats['total_runs']}")
    print(f"Total de genes seleccionados: {stats['total_genes_selected']}")
    print(f"Genes únicos seleccionados: {stats['unique_genes']}")
    print(f"Promedio de genes por ejecución: {stats['avg_genes_per_run']:.2f}")
    
    print(f"\nTop 5 genes más frecuentes en {algorithm}:")
    for gene, freq in stats['most_common_genes'].items():
        percentage = (freq / stats['total_runs']) * 100
        print(f"Gen {gene}: {freq} veces ({percentage:.1f}% de las ejecuciones)")
    
    print("\nEstadísticas por modelo:")
    for model, freq in stats['genes_by_model'].items():
        total_genes = sum(freq.values())
        unique_genes = len(freq)
        print(f"\n{model}:")
        print(f"  - Genes únicos: {unique_genes}")
        print(f"  - Total genes seleccionados: {total_genes}")
        print(f"  - Promedio genes por ejecución: {total_genes/stats['total_runs']:.2f}")

def analyze_all_algorithms(df):
    """
    Realiza el análisis completo para ambos algoritmos.
    """
    for algorithm in ['BACO', 'BPSO']:
        stats, fig = analyze_algorithm_genes(df, algorithm)
        print_algorithm_stats(stats, algorithm)
        plt.figure(fig.number)
        plt.show()

# Uso:
analyze_all_algorithms(result)
'''
#-------------------------------------------------------------------------#
#-------------------------------------------------------------------------#
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, VarianceThreshold
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score,StratifiedKFold, KFold
from sklearn import metrics as ms
from jmetal.util import load
import numpy as np

#baco = [0,538] 536,386,66
#bpso = [6,344] 440,2,399

# Cargar datos
data = load.huntington()
models_names, models = load.models()
X = data[0]
y = data[1]
clases = data[2]
i=-1

print(clases[0])
print(clases[2])
print(clases[6])
'''
# Inicializar KNN
knn = KNeighborsClassifier()

BACO_idx = [0,538,536,386,66]
BPSO_idx = [6,344,440,2,399]

sfs = SequentialFeatureSelector(models[i], n_features_to_select=10)
sfs.fit(X, y)
sfs_selected = sfs.get_support(indices=True)
print(f"SFS & {models_names[i]}")
print("Columnas seleccionadas:", len(sfs.get_support(indices=True)))
selected_features = [clases[i] for i in sfs.get_support(indices=True)]
# print(f"Features seleccionadas: {selected_features}")

# Mutual Information
mi = SelectKBest(score_func=mutual_info_classif, k=10)
mi.fit(X, y)
mi_selected = mi.get_support(indices=True)
print("Mutual Information:")
print("Columnas seleccionadas:", len(mi_selected))
mi_features = [clases[i] for i in mi_selected]
# print(f"Features seleccionadas: {mi_features}")

# Variance Threshold
variances = np.var(X, axis=0)
threshold = np.percentile(variances, 99)
var_thresh = VarianceThreshold(threshold=threshold)
X_var = var_thresh.fit_transform(X)
var_selected = var_thresh.get_support(indices=True)
print("Variance Threshold:")
print("Columnas seleccionadas:", len(var_selected))
var_features = [clases[i] for i in var_selected]
# print(f"Features seleccionadas: {var_features}")

# Random Forest Embedded
rf = RandomForestClassifier(max_depth=10)
rf.fit(X, y)
importances = rf.feature_importances_
rf_selected = np.argsort(importances)[-10:]  # Seleccionar las 10 características más importantes
print("Random Forest Embedded:")
print("Columnas seleccionadas:", len(rf_selected))
rf_features = [clases[i] for i in rf_selected]
# print(f"Features seleccionadas: {rf_features}")

# RESULTS

def evaluate_features(X, selected_features, y, n_splits=2):
    print(selected_features)
    acc_ls = []
    selected_features = np.flatnonzero(selected_features)
    X_selected = X[:, selected_features]
    alfa = 0.9
    for model in models:
        avg_acc = []
        kf = KFold(n_splits=2, shuffle=True, random_state=42)
        for train_index, test_index in kf.split(X_selected, y):
            Xtrain, Xtest = X_selected[train_index], X_selected[test_index]
            ytrain, ytest = y[train_index], y[test_index]
            model.fit(Xtrain, ytrain)
            y_pred = model.predict(Xtest)
            acc = ms.accuracy_score(ytest, y_pred)
            avg_acc.append(acc)
        avg_acc = np.mean(avg_acc)
        num_variables = len(selected_features)
        beta = 1 - alfa
        fitness = 1.0 - (num_variables/X.shape[1])
        fitness = (alfa * fitness) + (beta * avg_acc)
        acc_ls.append(fitness)
    return acc_ls

print("\nRF,SVM,KNN,AB,DT")
print(f"Accuracy BACO: {evaluate_features(X, BACO_idx, y)}")
print(f"Accuracy BPSO: {evaluate_features(X, BPSO_idx, y)}")


# SFS
BACO_in_sfs = [idx for idx in BACO_idx if idx in sfs_selected]
BPSO_in_sfs = [idx for idx in BPSO_idx if idx in sfs_selected]
print("\n¿Las características de BACO están en SFS?", BACO_in_sfs)
print("¿Las características de BPSO están en SFS?", BPSO_in_sfs)
print(f"Accuracy con SFS: {evaluate_features(X, sfs_selected, y)}")

# Mutual Information
BACO_in_mi = [idx for idx in BACO_idx if idx in mi_selected]
BPSO_in_mi = [idx for idx in BPSO_idx if idx in mi_selected]
print("\n¿Las características de BACO están en Mutual Information?", BACO_in_mi)
print("¿Las características de BPSO están en Mutual Information?", BPSO_in_mi)
print(f"Accuracy con Mutual Information: {evaluate_features(X, mi_selected, y)}")

# Variance Threshold
BACO_in_var = [idx for idx in BACO_idx if idx in var_selected]
BPSO_in_var = [idx for idx in BPSO_idx if idx in var_selected]
print("\n¿Las características de BACO están en Variance Threshold?", BACO_in_var)
print("¿Las características de BPSO están en Variance Threshold?", BPSO_in_var)
print(f"Accuracy con Variance Threshold: {evaluate_features(X, var_selected, y)}")

# Random Forest Embedded
BACO_in_rf = [idx for idx in BACO_idx if idx in rf_selected]
BPSO_in_rf = [idx for idx in BPSO_idx if idx in rf_selected]
print("\n¿Las características de BACO están en Random Forest Embedded?", BACO_in_rf)
print("¿Las características de BPSO están en Random Forest Embedded?", BPSO_in_rf)
print(f"Accuracy con Random Forest Embedded: {evaluate_features(X, rf_selected, y)}")


def compute_metrics(X, y, BACO_idx, BPSO_idx, sfs_selected, mi_selected, var_selected, rf_selected):
    """
    Calcula y devuelve un diccionario con las métricas de cada modelo basado en los métodos de selección.
    Además, genera un mapa de calor de la precisión.
    """
    results = {
        # "None FS": evaluate_features(X, list(range(X.shape[1])), y),
        "BACO": evaluate_features(X, BACO_idx, y),
        "BPSO": evaluate_features(X, BPSO_idx, y),
        "SFS": evaluate_features(X, sfs_selected, y),
        "MI": evaluate_features(X, mi_selected, y),
        "VT": evaluate_features(X, var_selected, y),
        "RF": evaluate_features(X, rf_selected, y)
    }

    # Convertir los resultados en un DataFrame para visualización con un mapa de calor
    heatmap_data = pd.DataFrame(results, index=models_names)

    # Crear mapa de calor
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, cmap="BuGn", fmt=".4f")#, cbar_kws={'label': 'Fitness'})
    plt.title("Fitness obtenido entre clasificadores y métodos de selección")
    plt.xlabel("Método de Selección de Características")
    plt.ylabel("Clasificadores")
    plt.savefig('fitness_fs_methods.pdf')
    plt.show()

    return results

# Llamada al método con los datos y las características seleccionadas
metrics_results = compute_metrics(
    X, y, 
    BACO_idx, BPSO_idx, 
    sfs_selected, 
    mi_selected, var_selected, 
    rf_selected
)

print("Resultados de precisión por modelo y método de selección:")
for method, accs in metrics_results.items():
    print(f"{method}: {dict(zip(models_names, accs))}")
'''