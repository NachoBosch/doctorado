from collections import Counter
from genes_collection import *
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.join(current_dir.replace('experimentos', ''))
sys.path.append(module_dir)
from jmetal.util import load  # Asegúrate de tener esta función o módulo para cargar los datos
import pandas as pd
from mygene import MyGeneInfo


# Cargar los datos y obtener los nombres de las columnas
data = load.huntington()

column_names = data[2]

# Combinar índices de todos los algoritmos en un solo diccionario
all_algorithms = {
    'BACO': BACO,
    'BALO': BALO,
    'BEO': BEO,
    'BGPC': BGPC,
    'BOA': BOA,
    'GWO': GWO,
    'SA': SA,
    'SS': SS,
    'UGA': UGA,
    'BPSO': BPSO,
    'DE': DE,
    'BHHO': BHHO,
}

# Contar las ocurrencias de índices por modelo y por algoritmo
def count_feature_occurrences(algorithms):
    feature_counts = {}
    for algo_name, models in algorithms.items():
        feature_counts[algo_name] = Counter(idx for model in models.values() for idx in model)
    return feature_counts

# Contar ocurrencias
feature_counts = count_feature_occurrences(all_algorithms)

# Mostrar las características más repetidas por algoritmo con sus nombres
for algo, counts in feature_counts.items():
    print(f"\n{algo} - Características más repetidas:")
    for idx, count in counts.most_common(5):  # Las 5 más frecuentes
        column_name = column_names[idx] if idx < len(column_names) else "Índice fuera de rango"
        print(f"Columna '{column_name}' (Índice {idx}): {count} veces")
'''

data = {
    "BACO": ["ENSG00000114948", "ENSG00000166833", "ENSG00000169641", "ENSG00000169282", "ENSG00000131095"],
    "BALO": ["ENSG00000120063", "ENSG00000163624", "ENSG00000111058", "ENSG00000164741", "ENSG00000165119"],
    "BEO": ["ENSG00000135439", "ENSG00000056736", "ENSG00000106610", "ENSG00000114948", "ENSG00000103056"],
    "BGPC": ["ENSG00000122435", "ENSG00000138083", "ENSG00000135387", "ENSG00000135365", "ENSG00000198216"],
    "BOA": ["ENSG00000124225", "ENSG00000184486", "ENSG00000168175", "ENSG00000140395", "ENSG00000155111"],
    "GWO": ["ENSG00000120875", "ENSG00000020922", "ENSG00000110925", "ENSG00000182134", "ENSG00000135643"],
    "SA": ["ENSG00000134440", "ENSG00000134294", "ENSG00000118579", "ENSG00000114948", "ENSG00000114796"],
    "SS": ["ENSG00000056736", "ENSG00000155313", "ENSG00000187079", "ENSG00000135387", "ENSG00000166501"],
    "UGA": ["ENSG00000164951", "ENSG00000134294", "ENSG00000174738", "ENSG00000090932", "ENSG00000143382"],
    "BPSO": ["ENSG00000130477", "ENSG00000103449", "ENSG00000135439", "ENSG00000145012", "ENSG00000162733"],
    "DE": ["ENSG00000100330", "ENSG00000196104", "ENSG00000134294", "ENSG00000165119", "ENSG00000134440"],
    "BHHO": ["ENSG00000114948", "ENSG00000106610", "ENSG00000184545", "ENSG00000157087", "ENSG00000084090"],
}


data_with_gene_names = {
    "BACO": ["ATXN2", "DNAJC6", "HTRA2", "PARK7", "UCHL1"],
    "BALO": ["HTT", "MTCH2", "NCOR1", "NPTX2", "PSEN1"],
    "BEO": ["APOE", "APP", "CNR1", "ATXN2", "BDNF"],
    "BGPC": ["AKT1", "CREB1", "GRIN2A", "GRIN2B", "MAPT"],
    "BOA": ["FOXP1", "ITPR1", "KCND3", "MAPK8", "PRNP"],
    "GWO": ["TOMM40", "UBC", "UCHL5", "UBE3A", "VDAC1"],
    "SA": ["ATG5", "ATG7", "BCL2", "ATXN2", "BAX"],
    "SS": ["APP", "DRD2", "GRIK2", "GRIN2A", "HTRA1"],
    "UGA": ["HTRA3", "ATG7", "MAP2", "PINK1", "VDAC2"],
    "BPSO": ["HDAC1", "IGF1", "APOE", "PPP3CA", "REST"],
    "DE": ["AKT3", "GRIK1", "ATG7", "PSEN1", "ATG5"],
    "BHHO": ["ATXN2", "CNR1", "SNCA", "SOD1", "TUBB3"],
}

# Crear un DataFrame a partir del diccionario
# df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))
df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data_with_gene_names.items()]))

# # Inicializar MyGeneInfo
# mg = MyGeneInfo()

# # Traducir ENSG a nombres comunes
# translated_genes = {}
# for column in df.columns:
#     gene_list = df[column].dropna().tolist()
#     query_result = mg.querymany(gene_list, scopes="ensembl.gene", fields="symbol", species="human")
#     translated_genes[column] = [gene.get('symbol', gene['query']) for gene in query_result]

# # Crear un nuevo DataFrame con los nombres comunes
# translated_df = pd.DataFrame(translated_genes)

# # Guardar el resultado en un nuevo CSV
# translated_filename = "translated_gene_selection.csv"
# translated_df.to_csv(translated_filename, index=False)
# print(f"Archivo CSV traducido guardado como {translated_filename}.")

translated_filename = "translated_gene_selection.csv"
df.to_csv(translated_filename, index=False)
print(f"Archivo CSV traducido guardado como {translated_filename}.")
'''