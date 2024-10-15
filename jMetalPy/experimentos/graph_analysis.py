import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import scikit_posthocs as sp
import seaborn as sns
sns.set(style='white',palette='bright')

bpso = pd.read_csv('../results/Resultados_toko/Resultados_BPSO/BPSO.csv')
sa = pd.read_csv('../results/Resultados_toko/Resultados_SA/SA.csv')
ss = pd.read_csv('../results/Resultados_toko/Resultados_SS/SS.csv')
baco = pd.read_csv('../results/Resultados_toko/Resultados_BACO/BACO.csv')
cga = pd.read_csv('../results/Resultados_toko/CGA.csv')
cga.rename(columns={'Vars':'Variables'},inplace=True)
de = pd.read_csv('../results/Resultados_toko/Resultados_DE/DE.csv')

# print(bpso.head(2))

#FITNESS 
def fitness_graph_bar(df1,df2,df3,df4,filepath):
    plt.figure(figsize=(10,10))
    
    df1['Algorithm'] = 'bpso'
    df2['Algorithm'] = 'sa'
    df3['Algorithm'] = 'ss'
    df4['Algorithm'] = 'baco'
    
    combined_df = pd.concat([df1, df2, df3,df4])

    # Plot the data using Seaborn
    ax = sns.barplot(x='Model', y='Fitness', 
                     hue='Algorithm', 
                     data=combined_df,
                     fill=True)
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.3f}', 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='center', 
                    xytext=(0, 5), 
                    textcoords='offset points')

    plt.minorticks_on()
    plt.tight_layout()
    plt.legend(loc='upper right', ncol=1)
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlabel('Models')
    plt.ylim(0,1.0)
    plt.ylabel('Fitness')
    plt.show()
    # plt.savefig(f"{filepath}.pdf")

# fitness_graph_bar(bpso,sa,ss,baco,'../results/Resultados_toko/models-fitness')

def fitness_graph_heatmap(df1, df2, df3, df4, df5, df6, filepath):
    plt.figure(figsize=(8, 6))

    algorithms = ['BPSO', 'BACO','DE','SA', 'SS','CGA']
    models = df1['Model'].unique()

    # Combine data into one array
    fitness_values = pd.concat([df1['Fitness'], 
                                df2['Fitness'], 
                                df3['Fitness'], 
                                df4['Fitness'],
                                df5['Fitness'],
                                df6['Fitness']], axis=1)
    fitness_values.columns = algorithms

    sns.heatmap(fitness_values, annot=True,fmt='0.3f', cmap='BuGn', cbar=True, xticklabels=algorithms, yticklabels=models)

    plt.xlabel('Algorithm')
    plt.ylabel('Model')
    plt.title('Fitness Value for Each Model and Algorithm')
    plt.tight_layout()
    plt.savefig(f"{filepath}.pdf")
    plt.show()

# fitness_graph_heatmap(bpso,baco,de,sa,ss,cga,'../results/Resultados_toko/models-fitness-heatmap')

def fitness_graph_dot(df1, df2, df3, df4, filepath):
    plt.figure(figsize=(10, 6))

    df1['Algorithm'] = 'bpso'
    df2['Algorithm'] = 'sa'
    df3['Algorithm'] = 'ss'
    df4['Algorithm'] = 'baco'

    combined_df = pd.concat([df1, df2, df3, df4])

    sns.stripplot(x='Model', y='Fitness', hue='Algorithm', data=combined_df, jitter=True, dodge=True)

    plt.ylim(0, 1.0)
    plt.xlabel('Models')
    plt.ylabel('Fitness')
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.show()

# fitness_graph_dot(bpso,sa,ss,baco,'../results/Resultados_toko/models-fitness')

###############################################################

def fitness_graph_plot(df1,df2,df3,df4,filepath):
    plt.figure(figsize=(10,10))
    
    df1['Algorithm'] = 'BPSO'
    df2['Algorithm'] = 'SA'
    df3['Algorithm'] = 'SS'
    df4['Algorithm'] = 'baco'
    
    combined_df = pd.concat([df1, df2, df3, df4])

    # Plot the data using Seaborn
    sns.lineplot(x='Model', y='Fitness', hue='Algorithm', data=combined_df)
    

    plt.minorticks_on()

    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlabel('Models')
    plt.ylim(0,1.0)
    plt.ylabel('Fitness')
    plt.show()
    # plt.savefig(f"{filepath}.pdf")

# fitness_graph_plot(bpso,sa,ss,'../results/Resultados_toko/AB/adaboost_fitness')


def vars_graph(df,filepath=None):
    plt.figure(figsize=(10,8))
    plt.plot(df['Alfa'],df['CX_0.7'],"r*-",label='CX_0.7')
    plt.plot(df['Alfa'],df['CX_0.8'],"b^-",label='CX_0.8')
    plt.plot(df['Alfa'],df['CX_0.9'],"g+-",label='CX_0.9')
    plt.minorticks_on()
    # for i in range(len(df)):
        # plt.text(df['Alfa'][i], df['CX_0.9'][i], f"{int(df['CX_0.9'][i])}", fontsize=9, color='green', ha='left')
    # max_value = df[df['Alfa'] == 0.9]['CX_0.9'].values[0]
    # plt.axhline(y=max_value, color='k', linestyle='--', linewidth=1)
    plt.legend()
    plt.grid()
    plt.xlabel('Alpha values')
    plt.ylabel('Variables Selected')
    plt.savefig(f"{filepath}.pdf")
    # plt.show()
# vars_graph(var_ab,'../results/Resultados_toko/AB/adaboost_vars')
# vars_graph(var_knn,'../results/Resultados_toko/KNN/knn_vars')
# vars_graph(var_rf,'../results/Resultados_toko/RF/randomforest_vars')
# vars_graph(var_svm,'../results/Resultados_toko/SVM/svm_vars')

def vars_graph_bar(df1,df2,df3,df4,filepath):
    plt.figure(figsize=(10,10))
    
    df1['Algorithm'] = 'bpso'
    df2['Algorithm'] = 'sa'
    df3['Algorithm'] = 'ss'
    df4['Algorithm'] = 'baco'
    
    combined_df = pd.concat([df1, df2, df3,df4])

    # Plot the data using Seaborn
    ax = sns.barplot(x='Model', y='Variables', 
                     hue='Algorithm', data=combined_df)
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{int(height)}', 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='center', 
                    xytext=(0, 5), 
                    textcoords='offset points')

    plt.minorticks_on()
    # plt.tight_layout()
    plt.legend(loc='upper right', ncol=1)
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlabel('Models')
    # plt.ylim(0,1.0)
    plt.ylabel('Variables')
    plt.show()
    # plt.savefig(f"{filepath}.pdf")

# vars_graph_bar(bpso,sa,ss,baco,'../results/Resultados_toko/models-var')

def vars_graph_heatmap(df1, df2, df3, df4,df5,df6, filepath):
    plt.figure(figsize=(8, 6))

    algorithms = ['BPSO', 'BACO','DE','SA', 'SS','CGA']
    models = df1['Model'].unique()

    # Combine data into one array
    variables_values = pd.concat([df1['Variables'], 
                                df2['Variables'], 
                                df3['Variables'], 
                                df4['Variables'],
                                df5['Variables'],
                                df6['Variables']], axis=1)
    variables_values.columns = algorithms

    sns.heatmap(variables_values, annot=True,fmt='d', cmap='BuGn_r', cbar=True, xticklabels=algorithms, yticklabels=models)

    plt.xlabel('Algorithm')
    plt.ylabel('Model')
    plt.title('Amount of Genes Selected for Each Model and Algorithm')
    plt.tight_layout()
    plt.savefig(f"{filepath}.pdf")
    plt.show()

# vars_graph_heatmap(bpso,baco,de,sa,ss,cga,'../results/Resultados_toko/models-vars-heatmap')

#################################################################
def norm_time(series):
    return (series - series.min()) / (series.max() - series.min())

def time_graph(df,filepath=None):

    df[['CX_0.7', 'CX_0.8', 'CX_0.9']] = df[['CX_0.7', 'CX_0.8', 'CX_0.9']].apply(norm_time)
    plt.figure(figsize=(10,8))
    plt.plot(df['Alfa'],df['CX_0.7'],"r*-",label='CX_0.7')
    plt.plot(df['Alfa'],df['CX_0.8'],"b^-",label='CX_0.8')
    plt.plot(df['Alfa'],df['CX_0.9'],"g+-",label='CX_0.9')
    plt.minorticks_on()
    # for i in range(len(df)):
        # plt.text(df['Alfa'][i], df['CX_0.9'][i], f"{df['CX_0.9'][i]:.3f}", fontsize=9, color='green', ha='left')
    # max_value = df[df['Alfa'] == 0.9]['CX_0.9'].values[0]
    # plt.axhline(y=max_value, color='k', linestyle='--', linewidth=1)
    plt.legend()
    plt.grid()
    plt.xlabel('Alpha values')
    plt.ylabel('Time Normalized')
    plt.savefig(f"{filepath}.pdf")
    # plt.show()

# time_graph(time_ab,'../results/Resultados_toko/AB/adaboost_time')
# time_graph(time_knn,'../results/Resultados_toko/KNN/knn_time')
# time_graph(time_rf,'../results/Resultados_toko/RF/randomforest_time')
# time_graph(time_svm,'../results/Resultados_toko/SVM/svm_time')

def fitness_best(df_rf,df_knn,df_svm,df_ab,filepath):
    plt.figure(figsize=(10,8))
    plt.plot(df_rf['Alfa'],df_rf['CX_0.9'],"r*-",label='RF')
    plt.plot(df_knn['Alfa'],df_knn['CX_0.9'],"g+-",label='KNN')
    plt.plot(df_svm['Alfa'],df_svm['CX_0.9'],"b^-",label='SVM')
    plt.plot(df_ab['Alfa'],df_ab['CX_0.9'],"m<-",label='AB')
    plt.minorticks_on()
    # for i in range(len(df)):
        # plt.text(df['Alfa'][i], df['CX_0.9'][i], f"{df['CX_0.9'][i]:.3f}", fontsize=9, color='green', ha='right')
    # max_value = df[df['Alfa'] == 0.9]['CX_0.9'].values[0]
    # plt.axhline(y=max_value, color='k', linestyle='--', linewidth=1)
    plt.legend()
    plt.grid()
    plt.xlabel('Alpha values')
    plt.ylabel('Fitness')
    plt.savefig(f"{filepath}.pdf")

# fitness_best(ft_rf,ft_knn,ft_svm,ft_ab,'../results/Resultados_toko/best_fitness_models')

def cross_graph(ft_rf,ft_knn,ft_svm,ft_ab,
                var_rf,var_knn,var_svm,var_ab,
                filepath):
    fig, ax1 = plt.subplots(figsize=(10,8))

    line1, = ax1.plot(ft_rf['Alfa'], ft_rf['CX_0.9'], 'ks--',mfc='white',linewidth=1.5,label='RF')
    line2, = ax1.plot(ft_knn['Alfa'],ft_knn['CX_0.9'],"kD--",mfc='white',linewidth=1.5,label='KNN')
    line3, = ax1.plot(ft_svm['Alfa'],ft_svm['CX_0.9'],"kX--",mfc='white',linewidth=1.5,label='SVM')
    line4, = ax1.plot(ft_ab['Alfa'],ft_ab['CX_0.9'],"k8--",mfc='white',linewidth=1.5,label='AB')
    plt.minorticks_on()
    ax1.set_xlabel('Alpha values')
    ax1.set_ylabel('Fitness')
    # plt.legend(loc='upper center')

    ax2 = ax1.twinx()
    line5, = ax2.plot(var_rf['Alfa'], var_rf['CX_0.9'], 'ks:',mfc='white')
    line6, = ax2.plot(var_knn['Alfa'], var_knn['CX_0.9'], "kD:",mfc='white')
    line7, = ax2.plot(var_svm['Alfa'], var_svm['CX_0.9'], "kX:",mfc='white')
    line8, = ax2.plot(var_ab['Alfa'], var_ab['CX_0.9'], "k8:",mfc='white')
    ax2.set_ylabel('Variables')

    lines = [line1, line2, line3, line4, line5, line6, line7, line8]
    labels = [line.get_label() for line in lines]
    
    fitness_legend = plt.Line2D([], [], color='black', linestyle='--')
    variables_legend = plt.Line2D([], [], color='black', linestyle=':')

    custom_lines = [fitness_legend, variables_legend]
    custom_labels = ['Fitness', 'Variables']
    
    print(custom_lines)
    # Combine all handles and labels
    combined_lines = lines + custom_lines
    combined_labels = labels + custom_labels
    
    ax1.legend(combined_lines, combined_labels, loc='upper center', ncol=3)
    

    plt.minorticks_on()
    # plt.show()
    plt.savefig(f"{filepath}.pdf")

# cross_graph(ft_rf,ft_knn,ft_svm,ft_ab,
#             var_rf,var_knn,var_svm,var_ab,
#             filepath="../results/Resultados_toko/fitness_vars_models")

def cross_data(df_ft,df_var,filepath):
    df_ft = df_ft[['Alfa','CX_0.9']]
    df_var = df_var[['Alfa','CX_0.9']]
    df = df_ft.merge(df_var, left_on='Alfa', right_on='Alfa' )
    df.rename(columns={'Alfa':'Alpha','CX_0.9_x':'Fitness',
               'CX_0.9_y':'Vars'},inplace=True)
    print(df)
    df.to_csv(f"{filepath}.csv")

# cross_data(ft_ab, var_ab,'../results/Resultados_toko/AB/ab_crosstab')

def cross_table(filepath):
    rf_ft = ft_rf[ft_rf['Alfa']==0.9]['CX_0.9'].item()
    knn_ft = ft_knn[ft_knn['Alfa']==0.9]['CX_0.9'].item()
    svm_ft = ft_svm[ft_svm['Alfa']==0.9]['CX_0.9'].item()
    ab_ft = ft_ab[ft_ab['Alfa']==0.9]['CX_0.9'].item()

    rf_var = var_rf[var_rf['Alfa']==0.9]['CX_0.9'].item()
    knn_var = var_knn[var_knn['Alfa']==0.9]['CX_0.9'].item()
    svm_var = var_svm[var_svm['Alfa']==0.9]['CX_0.9'].item()
    ab_var = var_ab[var_ab['Alfa']==0.9]['CX_0.9'].item()

    data = {
        'Model': ['RF', 'KNN', 'SVM', 'AB'],
        'Fitness': [rf_ft, knn_ft, svm_ft, ab_ft],
        'Vars': [rf_var, knn_var, svm_var, ab_var]
    }
    df = pd.DataFrame(data).set_index('Model')
    df['Vars']=df['Vars'].astype('int')
    print(df.T)
    df.to_csv(f"{filepath}.csv")
    
# cross_table(filepath='../results/Resultados_toko/cross_tab')

def test_dist_fitness(df_rf,df_knn,df_svm,df_ab,df_cga,df_de):
    print(f"BPSO:{stats.shapiro(df_rf['Fitness'])}")
    print(f"SA:{stats.shapiro(df_knn['Fitness'])}")
    print(f"SS:{stats.shapiro(df_svm['Fitness'])}")
    print(f"BACO:{stats.shapiro(df_ab['Fitness'])}")
    print(f"CGA:{stats.shapiro(df_cga['Fitness'])}")
    print(f"DE:{stats.shapiro(df_de['Fitness'])}")

    plt.figure(figsize=(10,8))

    df_rf['Fitness'].plot(kind='kde',label='BPSO',color='red')
    df_knn['Fitness'].plot(kind='kde',label='SA',color='green')
    df_svm['Fitness'].plot(kind='kde',label='SS',color='blue')
    df_ab['Fitness'].plot(kind='kde',label='BACO',color='magenta')
    df_cga['Fitness'].plot(kind='kde',label='CGA',color='purple')
    df_de['Fitness'].plot(kind='kde',label='DE',color='yellow')
    plt.grid()
    plt.legend()
    plt.show()

def test_dist_var(df_rf,df_knn,df_svm,df_ab,df_cga,df_de):
    print(f"BPSO:{stats.shapiro(df_rf['Variables'])}")
    print(f"SA:{stats.shapiro(df_knn['Variables'])}")
    print(f"SS:{stats.shapiro(df_svm['Variables'])}")
    print(f"BACO:{stats.shapiro(df_ab['Variables'])}")
    print(f"CGA:{stats.shapiro(df_cga['Variables'])}")
    print(f"DE:{stats.shapiro(df_de['Variables'])}")

def levene_test_fitness(df_rf,df_knn,df_svm,df_ab,df_cga,df_de):
    results = np.array([df_rf['Fitness'].to_list(),
               df_knn['Fitness'].to_list(),
               df_svm['Fitness'].to_list(),
               df_ab['Fitness'].to_list(),
               df_cga['Fitness'].to_list(),
               df_de['Fitness'].to_list()])
    stat, p = stats.levene(results[0],
                            results[1],
                            results[2],
                            results[3],
                            results[4],
                            results[5])
    print(f"Levene Fitness statistic: {stat} | p-value: {p}")

def levene_test_vars(df_rf,df_knn,df_svm,df_ab,df_cga,df_de):
    results = np.array([df_rf['Variables'].to_list(),
               df_knn['Variables'].to_list(),
               df_svm['Variables'].to_list(),
               df_ab['Variables'].to_list(),
               df_cga['Variables'].to_list(),
               df_de['Variables'].to_list()])
    stat, p = stats.levene(results[0],
                            results[1],
                            results[2],
                            results[3],
                            results[4],
                            results[5])
    print(f"Levene Variables statistic: {stat} | p-value: {p}")
    
def kruskal_wallis_fitness(*args,
                           names=None):
    results = np.array([df['Fitness'].to_list() for df in args]).T
    print(results)
    # stat, p = stats.kruskal(*results.T)
    stat, p = stats.kruskal(*[results[:, i] for i in range(results.shape[1])])
    print(f"Kruskal-Wallis Fitness statistic: {stat} | p-value: {p}")

    dunn = sp.posthoc_dunn(results.T, p_adjust='bonferroni')
    dunn.index = names
    dunn.columns = names
    print(f"Dunn Post-hoc:\n{dunn}")
    
    # Visualización
    plt.figure(figsize=(10, 8))
    sns.heatmap(dunn, annot=True, cmap='BuGn', center=0.05)
    plt.title("Dunn's test p-values (Bonferroni)")
    plt.show()

def kruskal_wallis_variables(*args, names=None):
    results = np.array([df['Variables'].to_list() for df in args]).T
    print(results)
    # stat, p = stats.kruskal(*results.T)
    stat, p = stats.kruskal(*[results[:, i] for i in range(results.shape[1])])
    print(f"Kruskal-Wallis Variables statistic: {stat} | p-value: {p}")

    dunn = sp.posthoc_dunn(results.T, p_adjust='bonferroni')
    dunn.index = names
    dunn.columns = names
    print(f"Dunn Post-hoc:\n{dunn}")
    
    # Visualización
    plt.figure(figsize=(10, 8))
    sns.heatmap(dunn, annot=True, cmap='BuGn', center=0.05)
    plt.title("Dunn's test p-values (Bonferroni)")
    plt.show()

def anova_ttest(df_rf,df_knn,df_svm,df_ab):
    rf_data = df_rf['CX_0.9'].to_list()
    knn_data = df_knn['CX_0.9'].to_list()
    svm_data = df_svm['CX_0.9'].to_list()
    ab_data = df_ab['CX_0.9'].to_list()

    print(f"T-Test (RF vs KNN): {stats.ttest_rel(rf_data, knn_data)}")
    print(f"T-Test (RF vs SVM): {stats.ttest_rel(rf_data, svm_data)}")
    print(f"T-Test (RF vs AB): {stats.ttest_rel(rf_data, ab_data)}")
    print(f"T-Test (KNN vs SVM): {stats.ttest_rel(knn_data, svm_data)}")
    print(f"T-Test (KNN vs AB): {stats.ttest_rel(knn_data, ab_data)}")
    print(f"T-Test (SVM vs AB): {stats.ttest_rel(svm_data, ab_data)}")
    stat, p = stats.f_oneway(rf_data, knn_data, svm_data, ab_data)
    print(f"ANOVA statistic: {stat} | p-value: {p}")

def test_dist_experiment(df):
    print(f"CX_0.7:{stats.shapiro(df['CX_0.7'])} | Kurtosis: {stats.kurtosis(df['CX_0.7'])} ")
    print(f"CX_0.8:{stats.shapiro(df['CX_0.8'])} | Kurtosis: {stats.kurtosis(df['CX_0.8'])}")
    print(f"CX_0.9:{stats.shapiro(df['CX_0.9'])}| Kurtosis: {stats.kurtosis(df['CX_0.9'])}")

    plt.figure(figsize=(10,8))

    df['CX_0.7'].plot(kind='kde',label='RF',color='red')
    df['CX_0.8'].plot(kind='kde',label='KNN',color='green')
    df['CX_0.9'].plot(kind='kde',label='SVM',color='blue')
    plt.grid()
    plt.legend()
    plt.show()

# -- STATS TESTS --

#NORMALITY
# test_dist_fitness(bpso,sa,ss,baco,cga,de)
# test_dist_var(bpso,sa,ss,baco,cga,de)

#HOMOSCEDASTICITY
# levene_test_fitness(bpso,sa,ss,baco,cga,de)
# levene_test_vars(bpso,sa,ss,baco,cga,de)

#COMPARISION
# kruskal_wallis_fitness(bpso,sa,ss,baco,cga,de,
#                        names=['BPSO', 'SA', 'SS', 'BACO', 'CGA', 'DE'])
# kruskal_wallis_variables(bpso,sa,ss,baco,cga,de,
#                          names=['BPSO', 'SA', 'SS', 'BACO', 'CGA', 'DE'])
# anova_ttest(ft_rf,ft_knn,ft_svm,ft_ab)
# kruskal_wallis(var_rf,var_knn,var_svm,var_ab)

# Variables Statistics test
# test_dist(var_rf,var_knn,var_svm,var_ab)
# anova_ttest(var_rf,var_knn,var_svm,var_ab)

#Experiment test
# test_dist_experiment(var_ab)
