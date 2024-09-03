import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import scikit_posthocs as sp

ft_ab = pd.read_csv('../results/Resultados_toko/AB/fitness_values_ab.csv')
var_ab = pd.read_csv('../results/Resultados_toko/AB/vars_selected_ab.csv')
time_ab = pd.read_csv('../results/Resultados_toko/AB/time_ab.csv')

ft_rf = pd.read_csv('../results/Resultados_toko/RF/fitness_values_rf.csv')
var_rf = pd.read_csv('../results/Resultados_toko/RF/vars_selected_rf.csv')
time_rf = pd.read_csv('../results/Resultados_toko/RF/time_rf.csv')

ft_knn = pd.read_csv('../results/Resultados_toko/KNN/fitness_values_knn.csv')
var_knn = pd.read_csv('../results/Resultados_toko/KNN/vars_selected_knn.csv')
time_knn = pd.read_csv('../results/Resultados_toko/KNN/time_knn.csv')

ft_svm = pd.read_csv('../results/Resultados_toko/SVM/fitness_values_svm.csv')
var_svm = pd.read_csv('../results/Resultados_toko/SVM/vars_selected_svm.csv')
time_svm = pd.read_csv('../results/Resultados_toko/SVM/time_svm.csv')

# print(ft_ab.head())

#FITNESS 
def fitness_graph(df,filepath):
    plt.figure(figsize=(10,8))
    plt.plot(df['Alfa'],df['CX_0.7'],"r*-",label='CX_0.7')
    plt.plot(df['Alfa'],df['CX_0.8'],"b^-",label='CX_0.8')
    plt.plot(df['Alfa'],df['CX_0.9'],"g+-",label='CX_0.9')
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

# fitness_graph(ft_ab,'../results/Resultados_toko/AB/adaboost_fitness')
# fitness_graph(ft_knn,'../results/Resultados_toko/KNN/knn_fitness')
# fitness_graph(ft_rf,'../results/Resultados_toko/RF/randomforest_fitness')
# fitness_graph(ft_svm,'../results/Resultados_toko/SVM/svm_fitness')

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

def test_dist(df_rf,df_knn,df_svm,df_ab):
    print(f"RF:{stats.shapiro(df_rf['CX_0.9'])} | Kurtosis: {stats.kurtosis(df_rf['CX_0.9'])} ")
    print(f"KNN:{stats.shapiro(df_knn['CX_0.9'])} | Kurtosis: {stats.kurtosis(df_knn['CX_0.9'])}")
    print(f"SVM:{stats.shapiro(df_svm['CX_0.9'])}| Kurtosis: {stats.kurtosis(df_svm['CX_0.9'])}")
    print(f"AB:{stats.shapiro(df_ab['CX_0.9'])}| Kurtosis: {stats.kurtosis(df_ab['CX_0.9'])}")

    plt.figure(figsize=(10,8))

    df_rf['CX_0.9'].plot(kind='kde',label='RF',color='red')
    df_knn['CX_0.9'].plot(kind='kde',label='KNN',color='green')
    df_svm['CX_0.9'].plot(kind='kde',label='SVM',color='blue')
    df_ab['CX_0.9'].plot(kind='kde',label='AB',color='magenta')
    plt.grid()
    plt.legend()
    plt.show()

def levene_test(df_rf,df_knn,df_svm,df_ab):
    results = np.array([df_rf['CX_0.9'].to_list(),
               df_knn['CX_0.9'].to_list(),
               df_svm['CX_0.9'].to_list(),
               df_ab['CX_0.9'].to_list()])
    stat, p = stats.levene(results[0],
                            results[1],
                            results[2],
                            results[3])
    print(f"Levene statistic: {stat} | p-value: {p}")

def kruskal_wallis(df_rf,df_knn,df_svm,df_ab):
    results = np.array([df_rf['CX_0.9'].to_list(),
               df_knn['CX_0.9'].to_list(),
               df_svm['CX_0.9'].to_list(),
               df_ab['CX_0.9'].to_list()])
    stat, p = stats.kruskal(results[0],
                            results[1],
                            results[2],
                            results[3])
    print(f"Kruskal-Wallis statistic: {stat} | p-value: {p}")

    dunn = sp.posthoc_dunn(results.T, p_adjust='bonferroni')
    print(f"Dunn Post-hoc:\n{dunn}")

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

# Fitness statistics test
# test_dist(ft_rf,ft_knn,ft_svm,ft_ab)
# test_dist(var_rf,var_knn,var_svm,var_ab)
# levene_test(ft_rf,ft_knn,ft_svm,ft_ab)
# levene_test(var_rf,var_knn,var_svm,var_ab)
# kruskal_wallis(ft_rf,ft_knn,ft_svm,ft_ab)
# anova_ttest(ft_rf,ft_knn,ft_svm,ft_ab)
# kruskal_wallis(var_rf,var_knn,var_svm,var_ab)

# Variables Statistics test
# test_dist(var_rf,var_knn,var_svm,var_ab)
anova_ttest(var_rf,var_knn,var_svm,var_ab)

#Experiment test
# test_dist_experiment(var_ab)
