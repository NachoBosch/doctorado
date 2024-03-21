library(affy)
library(affyPLM)
library(simpleaffy)
library(plyr)
library(limma)
library(ggplot2)
library(Biobase)

setwd('C:/Doctorado/doctorado/HD-microarray/HD/HG-U133A')
dataA <- ReadAffy()
qc.A <- qc(dataA)
plot(qc.A)

dataA.rma <- affy::rma(dataA)
qc.rma.A <- qc(dataA.rma)
plot(qc.rma.A)

hist(dataA, main = "Distribución de los datos crudos", xlab = "Valores", ylab = "Densidad")
hist(dataA.rma, main = "Distribución de los datos normalizados", xlab = "Valores", ylab = "Densidad")

setwd('C:/Doctorado/doctorado/Data/GSE3790-version2/GSE3790')
datos_full <- read.table("GSE3790.tsv", header = TRUE, sep = "\t")
datos_sin_gene <- subset(datos_full, select = -Gene)
hist(datos_sin_gene, main = "Distribución de los datos normalizados", xlab = "Valores", ylab = "Densidad")
par(mfrow = c(3, 3))
for (i in 1:9) {  # Cambia el rango según la cantidad de columnas que desees incluir
  hist(datos_sin_gene[, i], main = paste("Histograma de columna", i), xlab = "Valores", ylab = "Frecuencia")
}
par(mfrow = c(1, 1))

exprsA <- exprs(dataA.rma)
hist(exprsA, main = "Distribución de los datos de expresión", xlab = "Valores", ylab = "Densidad")

exprsA.raw <- exprs(dataA)
hist(exprsA.raw, main = "Distribución de los datos de expresión", xlab = "Valores", ylab = "Densidad")

