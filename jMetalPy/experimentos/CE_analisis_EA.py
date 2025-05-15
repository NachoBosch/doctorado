import matplotlib.pyplot as plt
import pandas as pd

# Datos extraídos de la imagen para Accuracy
accuracy_data = {
    'Alfa': [0.1, 0.5, 0.9],
    'SGA':   [0.7951, 0.7836, 0.7325],
    'uGA':   [0.7918, 0.7839, 0.7512],
    'CGA':   [0.7932, 0.7766, 0.7362],
}

fitness_data = {
    'Alfa': [0.1, 0.5, 0.9],
    'SGA':   [0.7702, 0.7444, 0.8412],
    'uGA':   [0.7646, 0.6979, 0.8246],
    'CGA':   [0.7668, 0.7256, 0.7889],
}

# Datos extraídos de la imagen para Genes
genes_data = {
    'Alfa': [0.1, 0.5, 0.9],
    'SGA':   [263, 171, 85],
    'uGA':   [278, 223, 97],
    'CGA':   [271, 188, 119],
}

# Convertimos a DataFrames
df_accuracy = pd.DataFrame(accuracy_data).set_index('Alfa')
df_fitness = pd.DataFrame(fitness_data).set_index('Alfa')
df_genes = pd.DataFrame(genes_data).set_index('Alfa')

# # Crear la figura y el eje
# fig, ax1 = plt.subplots(figsize=(12, 6))

# # Eje de la izquierda: Accuracy
# colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
# for i, algo in enumerate(df_accuracy.columns):
#     ax1.plot(df_accuracy.index, df_accuracy[algo], label=f'{algo}', color=colors[i], linestyle='-')
# ax1.set_xlabel('Alpha (α)')
# ax1.set_ylabel('Accuracy', color='black')
# ax1.tick_params(axis='y')

# # Eje de la derecha: Genes
# ax2 = ax1.twinx()
# for i, algo in enumerate(df_genes.columns):
#     ax2.plot(df_genes.index, df_genes[algo], color=colors[i], linestyle=':')
# ax2.set_ylabel('Genes', color='black')
# ax2.tick_params(axis='y')

# # Combinar leyendas
# lines_labels = [ax.get_legend_handles_labels() for ax in [ax1, ax2]]
# lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
# ax1.legend(lines, labels, loc='upper center', ncol=3)

# # plt.title('Accuracy & Genes vs Alfa')
# plt.grid(True)
# plt.tight_layout()
# fig.savefig('EAs_acc_genes.pdf')
# plt.show()

#--
# fig, ax1 = plt.subplots(figsize=(12, 6))

# # Eje de la izquierda: Fitness
# colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
# for i, algo in enumerate(df_fitness.columns):
#     ax1.plot(df_fitness.index, df_fitness[algo], label=f'{algo}', color=colors[i], linestyle='-')
# ax1.set_xlabel('Alpha (α)')
# ax1.set_ylabel('Fitness', color='black')
# ax1.tick_params(axis='y')

# # Eje de la derecha: Genes
# ax2 = ax1.twinx()
# for i, algo in enumerate(df_genes.columns):
#     ax2.plot(df_genes.index, df_genes[algo], color=colors[i], linestyle=':')
# ax2.set_ylabel('Genes', color='black')
# ax2.tick_params(axis='y')

# # Combinar leyendas
# lines_labels = [ax.get_legend_handles_labels() for ax in [ax1, ax2]]
# lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
# ax1.legend(lines, labels, loc='upper center', ncol=3)

# # plt.title('Fitness & Genes vs Alfa')
# plt.grid(True)
# plt.tight_layout()
# fig.savefig('EAs_fitness_genes.pdf')
# plt.show()

# #--
fig, ax1 = plt.subplots(figsize=(12, 6))

# Eje de la izquierda: Fitness
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
for i, algo in enumerate(df_fitness.columns):
    ax1.plot(df_fitness.index, df_fitness[algo], label=f'{algo}', color=colors[i], linestyle='-')
ax1.set_xlabel('Alpha (α)')
ax1.set_ylabel('Fitness', color='black')
ax1.tick_params(axis='y')

# Eje de la derecha: Accuracy
ax2 = ax1.twinx()
for i, algo in enumerate(df_accuracy.columns):
    ax2.plot(df_accuracy.index, df_accuracy[algo], color=colors[i], linestyle=':')
ax2.set_ylabel('Accuracy', color='black')
ax2.tick_params(axis='y')

# Combinar leyendas
lines_labels = [ax.get_legend_handles_labels() for ax in [ax1, ax2]]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
ax1.legend(lines, labels, loc='upper center', ncol=3)

# plt.title('Fitness & Genes vs Alfa')
plt.grid(True)
plt.tight_layout()
fig.savefig('EAs_acc_vs_fitness.pdf')
plt.show()