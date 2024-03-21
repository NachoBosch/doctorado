setwd('C:/Doctorado/doctorado/Microarray')
getwd()

#Paquetes para procesar microarrays
install.packages('BiocManager')
install.packages('Rtools')
BiocManager::install('affy')
BiocManager::install('limma')
BiocManager::install('affyPLM')
BiocManager::install("affyQCReport")
BiocManager::install(c("getopt", "oligo", "stringr", "reshape2", "data.table", "biomaRt"))
BiocManager::install("yaqcaffy")
BiocManager::install("oligoData")
BiocManager::install("Biobase",force=TRUE)
install.packages("yaqcaffy", dependencies = TRUE)
BiocManager::install("genefilter",force = TRUE)
#--------------------------------------
# 1. DATOS CRUDOS
#--------------------------------------
# Paso 1. Obtenemos los datos crudos de microarray usando el paquete affy
library("affy")

microarray.raw.data <- ReadAffy(verbose=TRUE)
microarray.raw.data

#Nombre del diseño de placa usada para las muestras
cdfName(microarray.raw.data)

#--------------------------------
# 2. CALIDAD
#--------------------------------
#Veremos con una imagen si los datos están bien (Control de calidad)
# No deberían observarse daños físicos, cómo manchas o rotura
# Esto debería replicarse para todas las muestras
library(mdqc)
library(affyPLM)
library(simpleaffy)
image(microarray.raw.data[,1], col=rainbow(100))
image(microarray.raw.data[,2], col=rainbow(100))

#Otro análisis de calidad es el Análisis QC

#
quality.analysis <- qc(microarray.raw.data)
plot(quality.analysis)
#Todo lo que aparece en azul es de buena calidad y lo que aparece en rojo
#es potencialmente problematico.
#La primer muestra es la que está en la base del gráfico
#Del análisis qc se obtienen dos porcentajes por cada muestra
# en el primero es el "porcentaje de detección" y representa el porcentaje de sondas
# en el microarray para las cuales se detectó un nivel de fluorescencia sustancial
# nunca da el 100% ya que no se expresa todo el genoma y todos los valores azules
# son los que son similares en todas las muestras ya que se asume que son de 
# condiciones comparables (aunque sean réplicas)
# el segundo valor se conoce como fluorescencia de fondo
# hay posiciones en el microarray que no se coloca ningun tipo de sondas pero si
# aparece una fluorescencia esto es una hibridación "inespecífica" de transcriptos
# al soporte fijo de microarray -> fondo (ruido del microarray)
# esto ultimo no resulta un problema ya que se puede eliminar el ruido en microarray
# Luego las relaciones 3 prima/5 prima para actina y gapdh nos puede decir
# si ha habido algun problema con la degradación del RNA y/o la sintesis de cDNA.
# En especifico esto se vería como un punto en rojo si cualquier cociente 
# fuera muchisimo mayor que 1 (siempre será mayor que 1).

#--------------------------------
# 3. PREPROCESAMIENTO
#--------------------------------
# En todo experimento siempre hay dos fuentes de variabilidad
# Variabilidad biológica (propia de lo que se investiga: tratamiento, mutación, etc) y 
# Variabilidad experimental (que es fuente de ruido del experimento mismo).
# para eliminar este ruido existe el algoritmo RMA (robust multi-array)
# este realiza una normalización de la fluorescencia de cada sonda y una estimación de
# los niveles de expresión de los genes representados por las sondas del microarray

# 1 - para comprobar que existe ruido de fondo vamos a generar una representación general
# de los niveles de los niveles de expresión en las muestras (boxplot - histogram)
boxplot(microarray.raw.data, las=2, col= rainbow(13), ylab="Fluorescence")

# Aplicamos correción de ruido de fondo
microarray.prepr <- rma(microarray.raw.data)
# este corrige el ruido de fondo, normaliza los datos y calcula el nivel de expresión 
# con una transformación de los niveles de expresión con log2
boxplot(microarray.prepr, las=2, col= rainbow(13), ylab="Fluorescence")

# Ahora queremos una matriz donde por filas tenga los niveles de expresión de los genes
# y por columnas las distintas muestras
exp.level <- exprs(microarray.prepr)
summary(exp.level)
dim(exp.level)
#En las filas tenemos los nombres de las sondas (código numérico seguido de at)

#--------------------------------
# 4. SELECCIÓN de genes que se expresan de forma diferencial
#--------------------------------
library(limma)#este paquete está creado para determinar genes que se expresan diferencialmente

#Para la selección de genes que se expresan de forma diferencial existen multitud de criterios
# los 3 mas comunes son:

# 1 . Fold change (factor de proporcionalidad): para esto se calcula el nivel de expresión media
# de haber control y tratamiento, se calcularía la razón de las medias en cada muestra para 
# cada gen. En microarray dada la normalización en log2, esto convierte la razón en una diferencia
# el umbral para genes activados sería (2,4,8) -> log2 (1,2,3) y para reprimidos (-1,-2,-3).
# Este método es para diseños experimentales con variabilidad muy muy baja, ya que no es un test estadístico

# 2. Estimación Estadística: contraste de hipótesis. Tener en cuenta que para tasas de testeo muy altas
# pueden existir una cantidad de falsos positivos muy alta
# por lo que se puede corregir el p-valor y convertirlo en un q-valor

# 3. Combinación de los dos anteriores.


