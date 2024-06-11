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
# new_solution.variables = [np.random.randint(0, 2) for _ in range(self.number_of_variables)]
# new_solution.objectives = [0 for _ in range(self.number_of_objectives)]


      # Genera un número aleatorio de características seleccionadas entre 10% y 90% del total
      # num_selected_features = np.random.randint(
      #     int(0.1 * self.number_of_variables), int(0.9 * self.number_of_variables) + 1
      # )
      # # Inicializa todas las variables a 0
      # new_solution.variables = [0] * self.number_of_variables
      # # Selecciona al azar las características que estarán activadas (1)
      # selected_indices = np.random.choice(range(self.number_of_variables), num_selected_features, replace=False)
      # for index in selected_indices:
      #   new_solution.variables[index] = 1