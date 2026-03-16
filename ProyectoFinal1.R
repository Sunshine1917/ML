# ======= PROYECTO FINAL ======= #
# Alejandro Juárez Rojas 
setwd("/Users/alejandrojuarezrojas/Downloads/ML")
getwd() #Cambiamos al directorio donde esta nuestro dataset

insurace <- read.csv("insurance.csv") 

# ======= Inspección inicial ======= #
str(insurace) #Vemos la esructura de datos, dimensión y variables
summary(insurace)

# Observamos si no contamos con valores NA en el data set
table(is.na(insurace))
colSums(is.na(insurace))

# Vemos si no tenemos valores duplicados
sum(duplicated(insurace))
insurace <- insurace[!duplicated(insurace),] #Se detecto un duplicado por lo que lo eliminamos

#Para detectar valores atípicos que puedan distorcionar el modelo 
par(mrfrow = c(2,2))
boxplot(insurace$age, main = "Age", col = "#2E75B6", ylab = "Value")
