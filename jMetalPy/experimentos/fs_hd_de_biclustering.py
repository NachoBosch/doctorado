import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
current_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.join(current_dir.replace('experimentos', ''))
sys.path.append(module_dir)

from jmetal.problems import Bic
from jmetal.algorithms.BPSO import BinaryPSOAlgorithm
from jmetal.core import crossover, mutation, selection
from jmetal.util.termination_criterion import StoppingByEvaluations
from results.results_local import Results
from jmetal.util.observer import PrintObjectivesObserver
from jmetal.util import load


class BiclusteringAnalysis:
    def __init__(self, data: np.ndarray, params: Dict[str, Any]):
        """
        Inicializa el análisis de biclustering
        
        Args:
            data: Matriz de datos de expresión génica
            params: Diccionario con los parámetros del algoritmo
        """
        self.data = data
        self.params = params
        self.problem = Bic.BiclusteringProblem(
            data=data,
            weights={
                'msr': 0.4,
                'volume': 0.3,
                'variance': 0.2,
                'overlap': 0.1
            },
            min_rows=2,
            min_cols=2
        )
        
    def run_analysis(self):
        """Ejecuta el análisis de biclustering y retorna los resultados"""
        # Configurar y ejecutar el algoritmo PSO
        algorithm = BinaryPSOAlgorithm(
            problem=self.problem,
            swarm_size=self.params['pobl'],
            inertia_weight=self.params['inertia_weight'],
            cognitive_coefficient=self.params['cognitive_coefficient'],
            social_coefficient=self.params['social_coefficient'],
            termination_criterion=StoppingByEvaluations(self.params['evals'])
        )
        
        # Registrar el observador para monitorear el progreso
        algorithm.observable.register(observer=PrintObjectivesObserver())
        
        # Ejecutar el algoritmo
        algorithm.run()
        return algorithm.result()
    
    def analyze_results(self, solution):
        """Analiza y visualiza los resultados del biclustering"""
        # Obtener información del bicluster
        bicluster_info = self.problem.get_bicluster_info(solution)
        
        # Imprimir resultados
        print("\n=== Resultados del Biclustering ===")
        print(f"Número de filas seleccionadas: {len(bicluster_info.rows)}")
        print(f"Número de columnas seleccionadas: {len(bicluster_info.columns)}")
        print(f"Mean Squared Residue (MSR): {bicluster_info.msr:.4f}")
        print(f"Volumen del bicluster: {bicluster_info.volume}")
        print(f"Varianza promedio: {bicluster_info.variance:.4f}")
        
        # Visualizar el bicluster
        self.visualize_bicluster(bicluster_info)
        
        return bicluster_info
    
    def visualize_bicluster(self, bicluster_info):
        """Visualiza el bicluster encontrado"""
        # Extraer el bicluster
        self.data = self.data.to_numpy()
        bicluster_data = self.data[np.ix_(bicluster_info.rows, bicluster_info.columns)]
        
        # Crear la visualización
        plt.figure(figsize=(12, 8))
        
        # Heatmap del bicluster
        sns.heatmap(bicluster_data, 
                   cmap='YlOrRd',
                   xticklabels=bicluster_info.columns,
                   yticklabels=bicluster_info.rows)
        
        plt.title('Bicluster Encontrado')
        plt.xlabel('Columnas')
        plt.ylabel('Filas')
        plt.show()

    def analyze_gene_patterns(self, bicluster_info, gene_names=None):
        """
        Analiza los patrones de genes en el bicluster
        
        Args:
            bicluster_info: Información del bicluster
            gene_names: Lista de nombres de genes (opcional)
        """
        # Obtener los índices de los genes seleccionados
        selected_genes = bicluster_info.rows
        
        # Extraer los datos de expresión de los genes seleccionados
        gene_expression = self.data[selected_genes, :]
        
        print("\n=== Análisis de Genes Significativos ===")
        print(f"Número total de genes seleccionados: {len(selected_genes)}")
        
        # Si tenemos nombres de genes, mostrarlos
        if gene_names is not None:
            print("\nGenes identificados:")
            for idx in selected_genes:
                print(f"Gen {gene_names[idx]}: ")
                print(f"  - Nivel medio de expresión: {np.mean(gene_expression[idx]):.4f}")
                print(f"  - Variación de expresión: {np.var(gene_expression[idx]):.4f}")
        else:
            print("\nÍndices de genes identificados:")
            for idx in selected_genes:
                print(f"Gen índice {idx}: ")
                print(f"  - Nivel medio de expresión: {np.mean(gene_expression[idx]):.4f}")
                print(f"  - Variación de expresión: {np.var(gene_expression[idx]):.4f}")
        
        # Análisis de correlación entre los genes seleccionados
        correlation_matrix = np.corrcoef(gene_expression)
        print("\nCorrelación media entre genes:", 
            np.mean(np.triu(correlation_matrix, k=1)))
        
        # Visualizar la correlación
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, 
                    annot=True, 
                    cmap='coolwarm',
                    center=0)
        plt.title('Correlación entre Genes Seleccionados')
        plt.show()
        
        return {
            'selected_genes': selected_genes,
            'expression_data': gene_expression,
            'correlation_matrix': correlation_matrix
        }

    def count_total_common_genes(self, solution):
        """
        Cuenta el número total de genes en común entre pares de condiciones.
        
        Args:
            solution: Solución del algoritmo de biclustering
        
        Returns:
            int: Número total de genes en común encontrados
        """
        # Obtener las filas (genes) y columnas (condiciones) seleccionadas
        rows = np.arange(self.data.shape[0])
        cols = np.arange(self.data.shape[1])
        selected_rows = rows[solution.variables[:self.data.shape[0]]]
        selected_cols = cols[solution.variables[self.data.shape[0]:]]
        
        total_common_genes = 0
        
        # Para cada par de condiciones
        for i in range(len(selected_cols)):
            for j in range(i + 1, len(selected_cols)):
                # Obtener los genes significativos para cada condición
                condition1 = selected_cols[i]
                condition2 = selected_cols[j]
                
                # Encontrar genes comunes entre estas dos condiciones
                common_genes = len(selected_rows)
                
                # Sumar al total
                total_common_genes += common_genes
                
                # Opcional: imprimir información detallada
                print(f"Entre condición {condition1} y condición {condition2}: {common_genes} genes en común")
        
        print(f"\nTotal de genes en común acumulado: {total_common_genes}")
        return total_common_genes

# Uso del código
if __name__ == "__main__":
    # Cargar datos
    data = load.huntington_bic()
    
    # Definir parámetros
    params = {
        'pobl': 100,
        'evals': 1000,
        'inertia_weight': 0.7,
        'cognitive_coefficient': 1.4,
        'social_coefficient': 1.4,
        'alfa': 0.9
    }
    
    # Crear y ejecutar el análisis
    analysis = BiclusteringAnalysis(data, params)
    
    # Ejecutar el algoritmo
    solution = analysis.run_analysis()
    
    # Analizar y visualizar resultados
    bicluster_info = analysis.analyze_results(solution)

    total_common = analysis.count_total_common_genes(solution)

    # gene_names = [] # Lista con los nombres de los genes
    # gene_analysis = analysis.analyze_gene_patterns(bicluster_info, gene_names)
    
    # Acceder a los resultados específicos
    print("\n=== Información adicional ===")
    print(f"Objectives: {solution.objectives}")
    print(f"Variables activas: {sum(solution.variables)}")




# data = load.huntington_bic()

# # print(data)

# #PARAMETERS
# params = {'pobl': 100,
#         'evals' : 1000,
#         'inertia_weight' :0.7,
#         'cognitive_coefficient': 1.4,
#         'social_coefficient':1.4,
#         'alfa':0.9
#         }

# #PROBLEM
# problem = Bic.BiclusteringProblem(data,params['alfa'])

# algorithm = BinaryPSOAlgorithm(
#     problem=problem,
#     swarm_size=params['pobl'],
#     inertia_weight=params['inertia_weight'],
#     cognitive_coefficient=params['cognitive_coefficient'],
#     social_coefficient=params['social_coefficient'],
#     termination_criterion=StoppingByEvaluations(params['evals'])
# )

# algorithm.observable.register(observer=PrintObjectivesObserver())
# algorithm.run()

# soluciones_ls = algorithm.result()
# print(f"Soluciones: {soluciones_ls}")
# objectives = soluciones_ls.objectives
# variables = soluciones_ls.variables

# print(f"Objectives: {objectives}")
# print(f"Variables: {sum(variables)}")

# RESULTS
# test = 'Biclustering'

# Results.results(algorithm,test,clases,params)

# algorithm.plot_fitness()
# algorithm.plot_min_variables()
# algorithm.save_csv(f'Results/Resultados_CGA/Resultados_nuevos/{test}.csv')