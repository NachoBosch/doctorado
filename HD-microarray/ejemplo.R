setwd('C:/Doctorado/doctorado/HD-microarray')
getwd()

data_dir <- file.path('Data','GSE3790')
data_file <- file.path(data_dir,'GSE3790.tsv')
metadata_file <- file.path(data_dir,'metadata_GSE3790.tsv')
grade_file <- file.path(data_dir,'df_HD_grade.csv')

file.exists(data_file)
file.exists(metadata_file)
file.exists(grade_file)

#LIBRARIES
library(limma)
library(magrittr)
library(ggplot2)
library(EnhancedVolcano)
set.seed(12345)

#DATA
metadata <- readr::read_tsv(metadata_file)
colnames(metadata)

grades <- readr::read_csv(grade_file)
colnames(grades)
names(grades)[names(grades) == "Samples"] <- "refinebio_accession_code"
grades$Grade <- ifelse(grades$Grade == "-", "Control", "Enfermedad")

expression_df <- readr::read_tsv(data_file) %>%
  tibble::column_to_rownames("Gene")

expression_df <- expression_df %>%
  dplyr::select(metadata$geo_accession)
all.equal(colnames(df), metadata$geo_accession)


#rename characteristics_ch1_genotype/variation
# and rename "overexpressing..." genotype to "CREB"
#metadata <- metadata %>%
#  dplyr::rename("genotype" = `characteristics_ch1_genotype/variation`) %>%
#  dplyr::mutate(
#    genotype = genotype %>%
#      forcats::fct_recode(CREB = "overexpressing the human CREB") %>%
#      forcats::fct_relevel("control")
#  )

#MERGE METADATA Y GRADES
# Realizar la combinación utilizando la función merge()
metadata <- merge(metadata, grades, by = "refinebio_accession_code", all.x = TRUE)
head(metadata)

#DESIGN MATRIX
des_mat <- model.matrix(~Grade, data = metadata)
head(des_mat)

#BAYES TEST
#We will use the lmFit() function from the limma package to test each gene
#for differential expression between the two groups using a linear model. 
#After fitting our data to the linear model, 
#in this example we apply empirical Bayes smoothing with the eBayses() function.

lmfit = lmFit(expression_df,design = des_mat)
ebays = eBayes(lmfit)

#STATS and Benjamini-Hochberg correction
stats_df <- topTable(ebays, number = nrow(expression_df)) %>%
  tibble::rownames_to_column("Gene")

#ENSG00000114948 primero de la lista
#ENSG00000170961 último de la lista
top_gene_df <- expression_df %>%
  dplyr::filter(rownames(.) == "ENSG00000056736") %>%
  t() %>%
  data.frame() %>%
  tibble::rownames_to_column("refinebio_accession_code") %>%
  dplyr::inner_join(dplyr::select(
    metadata,
    refinebio_accession_code,
    Grade
  ))

ggplot(top_gene_df, aes(x = Grade, y = ENSG00000056736, color = Grade)) +
  geom_jitter(width = 0.2, height = 0) +
  theme_classic()

#VOLCANO PLOT
EnhancedVolcano::EnhancedVolcano(stats_df,
                                 lab = stats_df$Gene,
                                 x = "logFC",
                                 y = "adj.P.Val"
)

volcano_plot <- EnhancedVolcano::EnhancedVolcano(stats_df,
                                                 lab = stats_df$Gene,
                                                 x = "logFC",
                                                 y = "adj.P.Val",
                                                 pCutoff = 0.01)
volcano_plot
