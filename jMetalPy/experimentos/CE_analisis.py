import matplotlib.pyplot as plt
import pandas as pd

# Datos extraídos de la imagen para Accuracy
accuracy_data = {
    'Alfa': [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
    'SGA':   [0.797, 0.795, 0.786, 0.748, 0.745, 0.730],
    'uGA':   [0.793, 0.787, 0.791, 0.761, 0.769, 0.772],
    'CGA':   [0.790, 0.786, 0.788, 0.752, 0.754, 0.745],
    'SS':    [0.771, 0.762, 0.747, 0.655, 0.658, 0.645],
    'DE':    [0.765, 0.750, 0.749, 0.693, 0.696, 0.676]
}

fitness_data = {
    'Alfa': [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
    'SGA':   [0.771, 0.751, 0.734, 0.796, 0.819, 0.838],
    'uGA':   [0.766, 0.740, 0.730, 0.747, 0.790, 0.833],
    'CGA':   [0.764, 0.744, 0.730, 0.752, 0.777, 0.788],
    'SS':    [0.744, 0.710, 0.684, 0.643, 0.626, 0.610],
    'DE':    [0.741, 0.715, 0.702, 0.645, 0.635, 0.630]
}

# Datos extraídos de la imagen para Genes
genes_data = {
    'Alfa': [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
    'SGA':   [268, 245, 224, 106, 94, 87],
    'uGA':   [276, 259, 238, 150, 118, 93],
    'CGA':   [270, 246, 234, 144, 126, 120],
    'SS':    [290, 288, 265, 210, 221, 228],
    'DE':    [271, 244, 236, 217, 220, 220]
}

# Convertimos a DataFrames
df_accuracy = pd.DataFrame(accuracy_data).set_index('Alfa')
df_fitness = pd.DataFrame(fitness_data).set_index('Alfa')
df_genes = pd.DataFrame(genes_data).set_index('Alfa')

# Crear la figura y el eje
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
# fig.savefig('EAs_fs_huntington.pdf')
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
# fig.savefig('EAs_fs_fitness_huntington.pdf')
# plt.show()

#--
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
fig.savefig('EAs_fs_acc_vs_fitness_huntington.pdf')
plt.show()