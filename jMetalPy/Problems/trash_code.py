#DATA
# df_hd = pd.read_csv('~/code/doctorado/Data/HD_dataset_full.csv')
# df_hd.rename(columns={'Unnamed: 0':'Samples'},inplace=True)
# df_hd['Grade'] = df_hd['Grade'].map({'-':'Control',
#                                      '0':'HD_0',
#                                      '1':'HD_1',
#                                      '2':'HD_2',
#                                      '3':'HD_3',
#                                      '4':'HD_4'})

#DEGs 
# degs = pd.read_csv('D:/Doctorado/doctorado/Data/genes_seleccionados_ebays.csv')
# degs = degs['Gene'].to_list()
# df_hd = df_hd[degs+['Samples','Grade']]

# new_variables = [list(np.random.randint(0, 2, size=1).tolist()[0] for _ in range(self.number_of_variables))]