library(affy)
library(affyPLM)
library(simpleaffy)
library(plyr)
library(limma)
library(ggplot2)
#library(oligo)
#library(oligoData)
library(Biobase)
#-------------------------------------------------------------------
# 1. Cargamos los datos y los exploramos
#------------------------------------------------------------------

# Leer datos HG-U133A
setwd('C:/Doctorado/doctorado/HD-microarray/HD/HG-U133A')

dataA <- ReadAffy()
cdfName(dataA)
dataA

#Observamos los datos crudos del grupo HD del set A
image(dataA[,1], col=rainbow(114))

#Análisis de calidad del grupo HD del set A
qc.A <- qc(dataA)
plot(qc.A)

# Leer datos HG-U133B
setwd('C:/Doctorado/doctorado/HD-microarray/HD/HG-U133B')

dataB <- ReadAffy()
cdfName(dataB)
dataB

#Observamos los datos crudos del grupo HD del set B
image(dataB[,1], col=rainbow(114))

#Análisis de calidad del grupo HD del set B
qc.B <- qc(dataB)
plot(qc.B)

#-------------------------------------------------------------------
# 2. RMA corrección de fondo, normalización y transformación log2
#------------------------------------------------------------------
# Normalizar los datos HG-U133A
boxplot(dataA, las=2, col= rainbow(13), xlab="Raw data A", ylab="Fluorescence")
dataA.rma <- rma(dataA)
boxplot(dataA.rma, las=2, col= rainbow(13), xlab="RMA data A", ylab="Fluorescence")
qc.rma.A <- qc(dataA.rma) 
plot(qc.rma.A)

# Normalizar los datos HG-U133B
boxplot(dataB, las=2, col= rainbow(13), xlab="Raw data B", ylab="Fluorescence")
dataB.rma <- rma(dataB)
boxplot(dataB.rma, las=2, col= rainbow(13), xlab="RMA data B", ylab="Fluorescence")

# Extraer los datos normalizados
# Ahora queremos una matriz donde por filas tenga los niveles de expresión de los genes
# y por columnas las distintas muestras
exprsA <- exprs(dataA.rma)
summary(exprsA)
dim(exprsA)

exprsB <- exprs(dataB.rma)
summary(exprsB)
dim(exprsB)

# Como ambos set del chip HU133 no tienen la misma cantidad de genes 
# primero tenemos que encontrar cuales son los que tienen en común
# para obtener la intersección de los nombres de genes. Esta forma no funciona.

# Probamos combinar los datos de ambos set
# permitiendo NA para genes que no están presentes en ambos conjuntos
combined.data <- merge(exprsA, exprsB, by = 0, all = TRUE)

# Elimina la primera columna que se genera automáticamente por merge
combined.data <- combined.data[, -1]

# Rellena los valores NA con ceros (o cualquier valor deseado)
#combined.data[is.na(combined.data)] <- 0
dim(combined.data)
boxplot(combined.data, las=2, col= rainbow(13), ylab="Fluorescence")

# Chequeamos valores nulos o nan
sum(is.na(combined.data))
filas_completas <- complete.cases(combined.data)
combined.data.clean <- combined.data[filas_completas, ]
sum(is.na(combined.data.clean)) #chequeamos existencia de valores nulos
dim(combined.data.clean)
boxplot(combined.data.clean, las=2, col= rainbow(228), ylab="Fluorescence")

#----------------CONTROL------------------------------------
setwd('C:/Doctorado/doctorado/HD-microarray/Control/HG-U133A')
control.A <- ReadAffy()
cdfName(control.A)
control.A
#Observamos los datos crudos del grupo CONTROL del set A
image(control.A[,1], col=rainbow(87))
#Análisis de calidad del grupo CONTROL del set A
qc.control.A <- qc(control.A)
plot(qc.control.A)

setwd('C:/Doctorado/doctorado/HD-microarray/Control/HG-U133B')
control.B <- ReadAffy()
cdfName(control.B)
control.B
#Observamos los datos crudos del grupo CONTROL del set B
image(control.B[,1], col=rainbow(89))
#Análisis de calidad del grupo CONTROL del set A
qc.control.B <- qc(control.B)
plot(qc.control.B)

# Normalizar los datos de control HG-U133A
boxplot(control.A, las=2, col= rainbow(13), xlab="Raw control A", ylab="Fluorescence")
control.A.rma <- rma(control.A)
boxplot(control.A.rma, las=2, col= rainbow(13), xlab="RMA control A", ylab="Fluorescence")

# Normalizar los datos de control HG-U133B
boxplot(control.B, las=2, col= rainbow(13), xlab="Raw control B", ylab="Fluorescence")
control.B.rma <- rma(control.B)
boxplot(control.B.rma, las=2, col= rainbow(13), xlab="RMA control B", ylab="Fluorescence")

# Extraer los datos normalizados
exprs.c.A <- exprs(control.A.rma)
summary(exprs.c.A)
dim(exprs.c.A)

exprs.c.B <- exprs(control.B.rma)
summary(exprs.c.B)
dim(exprs.c.B)

# Como ambos set del chip HU133 no tienen la misma cantidad de genes 
# primero tenemos que encontrar cuales son los que tienen en común
# para obtener la intersección de los nombres de genes. Esta forma no funciona.
# Probamos combinar los datos de ambos set
# permitiendo NA para genes que no están presentes en ambos conjuntos
combined.control <- merge(exprs.c.A, exprs.c.B, by = 0, all = TRUE)

# Elimina la primera columna que se genera automáticamente por merge
combined.control <- combined.control[, -1]

# Rellena los valores NA con ceros (o cualquier valor deseado)
#combined.data[is.na(combined.data)] <- 0
dim(combined.control)
boxplot(combined.control, las=2, col= rainbow(13), ylab="Fluorescence")

# Chequeamos valores nulos o nan
sum(is.na(combined.control))
filas_completas <- complete.cases(combined.control)
combined.control.clean <- combined.control[filas_completas, ]
sum(is.na(combined.control.clean)) #chequeamos existencia de valores nulos

boxplot(combined.control.clean, las=2, col= rainbow(228), ylab="Fluorescence")

#--------------------DATASET-------------
cat("Dimensión de expresión de A:",dim(exprsA))
cat("Dimensión de expresión de B:",dim(exprsB))
cat("Dimensión de expresión de control A:",dim(exprs.c.A))
cat("Dimensión de expresión de control B:",dim(exprs.c.B))

#114+87 = 201
#89+114 = 203

#Paper Hd:38+39+19+16, Control: 32+27+16+12
38+39+19+16
32+27+16+12

rownames_A <- rownames(exprsA)
rownames_c_A <- rownames(exprs.c.A)

identicals.rows <- rownames_A == rownames_c_A
identicals.rows

genes_a_visualizar <- exprsA[1:10000, ]  # Selecciona los primeros 100 genes
datos_para_grafico <- as.data.frame(t(genes_a_visualizar))
colnames(datos_para_grafico) <- paste0("Gen", 1:ncol(genes_a_visualizar))
ggplot(datos_para_grafico, aes(x = Gen1)) +
  geom_density(fill = "skyblue", color = "blue") +
  theme_minimal() +
  labs(x = "Expresión génica", y = "Densidad") +
  ggtitle("Distribuciones de expresión génica de U133A RMA")


#-------------------------------------------------------------------------------------
# 1. DISEÑO EXPERIMENTAL: entre conjunto HG-U133A con HD y control del mismo chip HG-U133A
#-------------------------------------------------------------------------------------

#factor que agrupa las muestras

dim(exprsA)
dim(exprs.c.A)
all.equal(rownames(exprsA),rownames(exprs.c.A))
n_hd <- ncol(dataA.rma)
n_c <- ncol(control.A.rma)

grupoA <- factor(c(rep("HD", n_hd), rep("Control", n_c)))
levels(grupoA)

design.A <- model.matrix(~ grupoA)
#design.A <- design.A[, -1]  # Eliminar la columna de intercepto
class(design.A)
dim(design.A)
#colnames(design.A) <- "HD"  # Nombrar la columna del diseño
#colnames(design.A) <- c("CONTROL", "HD")

expr.datos.A <- cbind(exprs.c.A, exprsA)

#expr.datos.A <- ExpressionSet(assayData = cbind(exprs.c.A, exprsA),
#                      phenoData = AnnotatedDataFrame(grupoA))

fit <- lmFit(expr.datos.A, design.A)
contrast <- makeContrasts(HD - CONTROL, levels = design.A)
fit.contrast <- contrasts.fit(fit, contrast)
fit.ebays <- eBayes(fit.contrast, trend = TRUE)

deg.A <- topTable(fit.ebays, number = Inf,sort.by = "P")
deg.A.sig <- subset(deg.A, P.Value < 0.001)

nrow(deg.A.sig)  # Obtener el número de genes significativos
#Misma cantidad de genes que el set original?
nrow(deg.A.sig) == nrow(exprsA)
 
#-------------------------------------------------------------------------------------
# 2. DISEÑO EXPERIMENTAL: entre conjunto HG-U133B con HD y control del mismo chip HG-U133B
#-------------------------------------------------------------------------------------

#factor que agrupa las muestras
all.equal(rownames(exprsB),rownames(exprs.c.B))
dim(control.B.rma)
dim(dataB.rma)

grupoB <- factor(c(rep("CONTROL", ncol(control.B.rma)), rep("HD", ncol(dataB.rma))))
levels(grupoB)

designB <- model.matrix(~grupoB)
colnames(designB) <- c("CONTROL", "HD")

expr.datos.B <- cbind(exprs.c.B, exprsB)

fit <- lmFit(expr.datos.B, designB)
contrast <- makeContrasts(HD - CONTROL, levels = designB)
fit.contrast <- contrasts.fit(fit, contrast)
fit.ebays <- eBayes(fit.contrast, trend = TRUE)

degB <- topTable(fit.ebays, number = Inf)
degB.sig <- subset(degB, P.Value < 0.001)

nrow(degB.sig)  # Obtener el número de genes significativos
#Misma cantidad de genes que el set original?
nrow(degB.sig) == nrow(exprsB)


















