# ======= PROYECTO FINAL ======= #
# Alejandro Juárez Rojas 

library(tidyverse) #Cargamos la libreria para una mejor visualización de los datos

getwd() #Cambiamos al directorio donde esta nuestro dataset
setwd("/Users/alejandrojuarezrojas/Downloads/ML")

insurance_data <- read.csv("insurance.csv")

#Visualizamos la estructura general del dataset y sus estadisticas
str(insurance_data)
summary(insurance_data)

head(insurance_data)
# Visualizando la tabla de valores categoricas 
table(insurance_data$sex)
table(insurance_data$smoker)
table(insurance_data$region)

# ======== ANALIZANDO DATOS ======= #
library(ggplot2)

# Distribuciones individuales
ggplot(insurance_data, aes(x = charges)) +
  geom_histogram(binwidth = 2000, fill = "#0073C2FF", color = "black") +
  labs(title = "Distribución de charges", x = "Charges (USD)", y = "Frecuencia")

ggplot(insurance_data, aes(x = age)) +
  geom_histogram(binwidth = 5, fill = "#0073C2FF", color = "black") +
  labs(title = "Distribución de edad", x = "Edad", y = "Frecuencia")

ggplot(insurance_data, aes(x = bmi)) +
  geom_histogram(binwidth = 2, fill = "#868686FF", color = "black") +
  labs(title = "Distribución de BMI", x = "BMI", y = "Frecuencia")

ggplot(insurance_data, aes(x = factor(children))) +
  geom_bar(fill = "#EFC000FF") +
  labs(title = "Distribución de número de hijos",
       x = "Número de hijos", y = "Frecuencia")

ggplot(insurance_data, aes(x = smoker)) +
  geom_bar(fill = "#CD534CFF") +
  labs(title = "Distribución de fumadores", x = "Fumador", y = "Frecuencia")

ggplot(insurance_data, aes(x = sex)) +
  geom_bar(fill = "#0073C2FF") +
  labs(title = "Distribución por sexo", x = "Sexo", y = "Frecuencia")

ggplot(insurance_data, aes(x = region)) +
  geom_bar(fill = "#868686FF") +
  labs(title = "Distribución por región", x = "Región", y = "Frecuencia")

# Relación de cada variable con charges
ggplot(insurance_data, aes(x = smoker, y = charges)) +
  geom_boxplot(fill = "#EFC000FF") +
  labs(title = "Charges por nivel de tabaquismo",
       x = "Fumador", y = "Charges (USD)")

ggplot(insurance_data, aes(x = sex, y = charges)) +
  geom_boxplot(fill = "#868686FF") +
  labs(title = "Charges por sexo", x = "Sexo", y = "Charges (USD)")

ggplot(insurance_data, aes(x = region, y = charges)) +
  geom_boxplot(fill = "#CD534CFF") +
  labs(title = "Charges por región", x = "Región", y = "Charges (USD)")

ggplot(insurance_data, aes(x = factor(children), y = charges)) +
  geom_boxplot(fill = "#0073C2FF") +
  labs(title = "Charges por número de hijos",
       x = "Número de hijos", y = "Charges (USD)")

ggplot(insurance_data, aes(x = age, y = charges, color = smoker)) +
  geom_point(alpha = 0.5) +
  labs(title = "Edad vs Charges", x = "Edad", y = "Charges (USD)",
       color = "Fumador")

ggplot(insurance_data, aes(x = bmi, y = charges, color = smoker)) +
  geom_point(alpha = 0.5) +
  labs(title = "BMI vs Charges", x = "BMI", y = "Charges (USD)",
       color = "Fumador")

# ======== TRANSFORMAR ======== #
insurance_data$sex    <- ifelse(insurance_data$sex == "female", 1, 0)
insurance_data$smoker <- ifelse(insurance_data$smoker == "yes", 1, 0)

insurance_data <- insurance_data %>%
  mutate(region_northeast = ifelse(region == "northeast", 1, 0),
         region_northwest = ifelse(region == "northwest", 1, 0),
         region_southeast = ifelse(region == "southeast", 1, 0)) %>%
  select(-region)

sum(is.na(insurance_data))
sum(duplicated(insurance_data))
insurance_data <- insurance_data[!duplicated(insurance_data), ]

range(insurance_data$age)
range(insurance_data$bmi)
range(insurance_data$children)
range(insurance_data$charges)

insurance_data_norm <- insurance_data %>%
  mutate(across(c(age, bmi, children, charges), scale))

str(insurance_data_norm)

# Matriz de correlación COMPLETA con todas las variables ya numéricas
insurance_data %>%
  select(age, bmi, children, sex, smoker,
         region_northeast, region_northwest, region_southeast,
         charges) %>%
  cor() %>%
  round(2)

#Complot matriz de correlación
library(corrplot)

cor_matrix <- insurance_data %>%
  select(age, bmi, children, sex, smoker,
         region_northeast, region_northwest, region_southeast,
         charges) %>%
  cor() %>%
  round(2)

corrplot(cor_matrix,
         method      = "color",
         addCoef.col = "black",
         tl.col      = "black",
         tl.srt      = 45,
         title       = "Heatmap de correlación — insurance",
         mar         = c(0, 0, 1, 0))


# ======== LIMPIEZA DE DATOS ======== #
library(car)
modelo_vif <- lm(charges ~ age + bmi + children + sex + smoker +
                   region_northeast + region_northwest + region_southeast,
                 data = insurance_data)
vif(modelo_vif)

# ======== ENTRENAMIENTO DEL MODELO ======== #
library(caret)
library(rpart)
library(rpart.plot)

#División del dataset 70% entrenamiento, 30% prueba
set.seed(123)
trainIndex <- createDataPartition(insurance_data$charges, p = 0.7, list = FALSE)
train_set  <- insurance_data[trainIndex, ]
test_set   <- insurance_data[-trainIndex, ]

modelo_lineal <- lm(charges ~ age + bmi + children + sex + smoker +
                      region_northeast + region_northwest + region_southeast,
                    data = train_set)
summary(modelo_lineal)

# Predicciones
pred_lineal <- predict(modelo_lineal, newdata = test_set)

# Métricas
mae_lineal  <- mean(abs(pred_lineal - test_set$charges))
rmse_lineal <- sqrt(mean((pred_lineal - test_set$charges)^2))
r2_lineal   <- cor(pred_lineal, test_set$charges)^2


cat("\n--- Regresión Lineal ---\n")
cat("MAE: ", round(mae_lineal, 2), "\n")
cat("RMSE:", round(rmse_lineal, 2), "\n")
cat("R²:  ", round(r2_lineal, 4), "\n")

# Gráfico real vs predicho
ggplot(data.frame(real = test_set$charges, predicho = pred_lineal),
       aes(x = real, y = predicho)) +
  geom_point(alpha = 0.4, color = "#0073C2FF") +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Regresión lineal — Real vs Predicho",
       x = "Charges real (USD)", y = "Charges predicho (USD)")

modelo_arbol <- rpart(charges ~ age + bmi + children + sex + smoker +
                        region_northeast + region_northwest + region_southeast,
                      data   = train_set,
                      method = "anova",
                      control = rpart.control(maxdepth = 5, minsplit = 20))
# Visualizar el árbol
rpart.plot(modelo_arbol,
           type  = 4,
           extra = 101,
           under = TRUE,
           main  = "Árbol de decisión — charges")

# Importancia de variables
cat("\nImportancia de variables:\n")
print(round(modelo_arbol$variable.importance, 2))

# Predicciones
pred_arbol <- predict(modelo_arbol, newdata = test_set)

# Métricas
mae_arbol  <- mean(abs(pred_arbol - test_set$charges))
rmse_arbol <- sqrt(mean((pred_arbol - test_set$charges)^2))
r2_arbol   <- cor(pred_arbol, test_set$charges)^2
cat("\n--- Árbol de Decisión ---\n")
cat("MAE: ", round(mae_arbol, 2), "\n")
cat("RMSE:", round(rmse_arbol, 2), "\n")
cat("R²:  ", round(r2_arbol, 4), "\n")


# Gráfico real vs predicho
ggplot(data.frame(real = test_set$charges, predicho = pred_arbol),
       aes(x = real, y = predicho)) +
  geom_point(alpha = 0.4, color = "#EFC000FF") +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Árbol de decisión — Real vs Predicho",
       x = "Charges real (USD)", y = "Charges predicho (USD)")

cat("\n===== COMPARACIÓN DE MODELOS =====\n")
cat(sprintf("%-25s %10s %10s %10s\n", "Modelo", "MAE", "RMSE", "R²"))
cat(sprintf("%-25s %10.2f %10.2f %10.4f\n", "Regresión Lineal",
            mae_lineal, rmse_lineal, r2_lineal))
cat(sprintf("%-25s %10.2f %10.2f %10.4f\n", "Árbol de Decisión",
            mae_arbol, rmse_arbol, r2_arbol))

# ======= MEJORA DEL PROCESO ======= #
dim(insurance_data)
str(insurance_data)

insurance_data <- insurance_data %>%
  mutate(
    age2         = age^2,
    smoker_bmi   = smoker * bmi,
    smoker_age   = smoker * age,
    bmi_obese    = ifelse(bmi >= 30, 1, 0),
    smoker_obese = smoker * bmi_obese
  )

# Verificar que quedaron bien
insurance_data %>%
  select(age, age2, bmi, bmi_obese, smoker, 
         smoker_bmi, smoker_age, smoker_obese) %>%
  head(10)

insurance_data %>%
  select(age2, smoker_bmi, smoker_age, bmi_obese, smoker_obese) %>%
  summary()

# Gráfico 1 smoker_bmi vs charges
ggplot(insurance_data, aes(x = smoker_bmi, y = charges, 
                           color = factor(smoker))) +
  geom_point(alpha = 0.4) +
  labs(title = "Interacción smoker×BMI vs Charges",
       x = "smoker × BMI", y = "Charges (USD)",
       color = "Fumador") +
  scale_color_manual(values = c("0" = "#0073C2FF", "1" = "#CD534CFF"))

# Gráfico 2: age2 vs charges
ggplot(insurance_data, aes(x = age2, y = charges)) +
  geom_point(alpha = 0.3, color = "#868686FF") +
  geom_smooth(method = "lm", color = "red") +
  labs(title = "age² vs Charges", x = "Edad²", y = "Charges (USD)")

cor_matrix_v2 <- insurance_data %>%
  select(age, age2, bmi, bmi_obese, children, sex, smoker,
         smoker_bmi, smoker_age, smoker_obese,
         region_northeast, region_northwest, region_southeast,
         charges) %>%
  cor() %>%
  round(2)

# Ranking de correlación con charges
sort(cor_matrix_v2[, "charges"], decreasing = TRUE)

corrplot(cor_matrix_v2,
         method      = "color",
         addCoef.col = "black",
         tl.col      = "black",
         tl.srt      = 45,
         title       = "Matriz de correlación — variables originales y derivadas",
         mar         = c(0, 0, 1, 0))

set.seed(123)
trainIndex <- createDataPartition(insurance_data$charges, p = 0.7, list = FALSE)
train_set  <- insurance_data[trainIndex, ]
test_set   <- insurance_data[-trainIndex, ]

modelo_lineal_v2 <- lm(charges ~ age + age2 + bmi + children + sex + smoker +
                         smoker_bmi + smoker_age + smoker_obese + bmi_obese +
                         region_northeast + region_northwest + region_southeast,
                       data = train_set)

summary(modelo_lineal_v2)

pred_lineal_v2 <- predict(modelo_lineal_v2, newdata = test_set)

mae_lineal_v2  <- mean(abs(pred_lineal_v2 - test_set$charges))
rmse_lineal_v2 <- sqrt(mean((pred_lineal_v2 - test_set$charges)^2))
r2_lineal_v2   <- cor(pred_lineal_v2, test_set$charges)^2

cat("\n===== COMPARACIÓN =====\n")
cat(sprintf("%-30s MAE=%8.2f  RMSE=%8.2f  R²=%.4f\n",
            "Lineal original",
            mae_lineal, rmse_lineal, r2_lineal))
cat(sprintf("%-30s MAE=%8.2f  RMSE=%8.2f  R²=%.4f\n",
            "Lineal v2 con features",
            mae_lineal_v2, rmse_lineal_v2, r2_lineal_v2))

#======= ETAPA DE LASSO Y RIDGET ========#
install.packages("glmnet")
library(glmnet)

# glmnet NO acepta data.frame, necesita matrices
X_train <- model.matrix(charges ~ age + age2 + bmi + children + sex + 
                          smoker + smoker_bmi + smoker_age + 
                          smoker_obese + bmi_obese +
                          region_northeast + region_northwest + 
                          region_southeast, 
                        data = train_set)[, -1]  # -1 elimina el intercepto

y_train <- train_set$charges

X_test <- model.matrix(charges ~ age + age2 + bmi + children + sex + 
                         smoker + smoker_bmi + smoker_age +
                         smoker_obese + bmi_obese +
                         region_northeast + region_northwest + 
                         region_southeast, 
                       data = test_set)[, -1]

y_test <- test_set$charges

# Verificar dimensiones
dim(X_train)  # debe ser ~937 x 13
dim(X_test)   # debe ser ~400 x 13

# Ridge 
set.seed(123)
cv_ridge <- cv.glmnet(X_train, y_train, 
                      alpha  = 0,      # 0 = Ridge
                      nfolds = 10)

# Ver el lambda óptimo
cat("Lambda óptimo Ridge:", round(cv_ridge$lambda.min, 2), "\n")

# Gráfico de validación cruzada
plot(cv_ridge)
title("Ridge — CV para selección de lambda", line = 3)

# Predicciones
pred_ridge <- as.vector(predict(cv_ridge, 
                                s      = "lambda.min", 
                                newx   = X_test))

mae_ridge  <- mean(abs(pred_ridge - y_test))
rmse_ridge <- sqrt(mean((pred_ridge - y_test)^2))
r2_ridge   <- cor(pred_ridge, y_test)^2

cat("\n--- Ridge ---\n")
cat("MAE: ", round(mae_ridge,  2), "\n")
cat("RMSE:", round(rmse_ridge, 2), "\n")
cat("R²:  ", round(r2_ridge,   4), "\n")

#LASSO 
set.seed(123)
cv_lasso <- cv.glmnet(X_train, y_train, 
                      alpha  = 1,      # 1 = Lasso
                      nfolds = 10)

cat("Lambda óptimo Lasso:", round(cv_lasso$lambda.min, 2), "\n")

plot(cv_lasso)
title("Lasso — CV para selección de lambda", line = 3)

# Predicciones
pred_lasso <- as.vector(predict(cv_lasso, 
                                s    = "lambda.min", 
                                newx = X_test))

mae_lasso  <- mean(abs(pred_lasso - y_test))
rmse_lasso <- sqrt(mean((pred_lasso - y_test)^2))
r2_lasso   <- cor(pred_lasso, y_test)^2

cat("\n--- Lasso ---\n")
cat("MAE: ", round(mae_lasso,  2), "\n")
cat("RMSE:", round(rmse_lasso, 2), "\n")
cat("R²:  ", round(r2_lasso,   4), "\n")

# Coeficientes de ambos modelos lado a lado
coefs <- cbind(
  Ridge = round(as.vector(coef(cv_ridge, s = "lambda.min")), 4),
  Lasso = round(as.vector(coef(cv_lasso, s = "lambda.min")), 4)
)
rownames(coefs) <- rownames(coef(cv_ridge, s = "lambda.min"))
print(coefs)

cat("\n===== COMPARACIÓN ACUMULADA =====\n")
cat(sprintf("%-30s MAE=%8.2f  RMSE=%8.2f  R²=%.4f\n",
            "Lineal original",      mae_lineal,    rmse_lineal,    r2_lineal))
cat(sprintf("%-30s MAE=%8.2f  RMSE=%8.2f  R²=%.4f\n",
            "Lineal v2 + features", mae_lineal_v2, rmse_lineal_v2, r2_lineal_v2))
cat(sprintf("%-30s MAE=%8.2f  RMSE=%8.2f  R²=%.4f\n",
            "Ridge",                mae_ridge,     rmse_ridge,     r2_ridge))
cat(sprintf("%-30s MAE=%8.2f  RMSE=%8.2f  R²=%.4f\n",
            "Lasso",                mae_lasso,     rmse_lasso,     r2_lasso))


#======= Adaptive Lasso =======#

# ======= Paso 1: Estandarizar X manualmente =======
medias <- colMeans(X_train)
desv   <- apply(X_train, 2, sd)

X_train_s <- scale(X_train, center = medias, scale = desv)
X_test_s  <- scale(X_test,  center = medias, scale = desv)

# Verificar
dim(X_train_s)  # 937 x 13
dim(X_test_s)   # 400 x 13

# ======= datos estandarizados =======
set.seed(123)
cv_ridge_s <- cv.glmnet(X_train_s, y_train, 
                        alpha  = 0, 
                        nfolds = 10)

cat("Lambda Ridge estandarizado:", round(cv_ridge_s$lambda.min, 2), "\n")

# Pesos desde coeficientes estandarizados
coef_ridge_s  <- as.matrix(coef(cv_ridge_s, s = "lambda.min"))
# Define gamma (g) before using it
g <- 1

w_adapt_s     <- 1 / (abs(coef_ridge_s[-1, ]) ^ g)

# Ahora los pesos deben tener más sentido
cat("\nPesos Adaptive Lasso (estandarizados):\n")
round(sort(w_adapt_s, decreasing = TRUE), 4)

# ======= Adaptive Lasso corregido =======
set.seed(123)
cv_alasso_s <- cv.glmnet(X_train_s, y_train,
                         alpha          = 1,
                         penalty.factor = w_adapt_s,
                         nfolds         = 10)

cat("\nLambda óptimo ALasso corregido:", round(cv_alasso_s$lambda.min, 2), "\n")

pred_alasso_s <- as.vector(predict(cv_alasso_s, 
                                   s    = "lambda.min", 
                                   newx = X_test_s))

mae_alasso_s  <- mean(abs(pred_alasso_s - y_test))
rmse_alasso_s <- sqrt(mean((pred_alasso_s - y_test)^2))
r2_alasso_s   <- cor(pred_alasso_s, y_test)^2

cat("\n--- Adaptive Lasso corregido ---\n")
cat("MAE: ", round(mae_alasso_s,  2), "\n")
cat("RMSE:", round(rmse_alasso_s, 2), "\n")
cat("R²:  ", round(r2_alasso_s,   4), "\n")

# ======= Tabla de coeficientes corregida =======
coefs_final <- cbind(
  Ridge  = round(as.vector(coef(cv_ridge,   s = "lambda.min")), 4),
  Lasso  = round(as.vector(coef(cv_lasso,   s = "lambda.min")), 4),
  ALasso = round(as.vector(coef(cv_alasso_s, s = "lambda.min")), 4)
)
rownames(coefs_final) <- rownames(coef(cv_ridge, s = "lambda.min"))
print(coefs_final)

#=======Paso 5: Tabla acumulada =======
cat("\n===== COMPARACIÓN ACUMULADA =====\n")
cat(sprintf("%-30s MAE=%8.2f  RMSE=%8.2f  R²=%.4f\n",
            "Lineal original",        mae_lineal,    rmse_lineal,    r2_lineal))
cat(sprintf("%-30s MAE=%8.2f  RMSE=%8.2f  R²=%.4f\n",
            "Lineal v2 + features",   mae_lineal_v2, rmse_lineal_v2, r2_lineal_v2))
cat(sprintf("%-30s MAE=%8.2f  RMSE=%8.2f  R²=%.4f\n",
            "Ridge",                  mae_ridge,     rmse_ridge,     r2_ridge))
cat(sprintf("%-30s MAE=%8.2f  RMSE=%8.2f  R²=%.4f\n",
            "Lasso",                  mae_lasso,     rmse_lasso,     r2_lasso))
cat(sprintf("%-30s MAE=%8.2f  RMSE=%8.2f  R²=%.4f\n",
            "ALasso corregido",       mae_alasso_s,  rmse_alasso_s,  r2_alasso_s))

# ======== Random Forest ======== #
install.packages("randomForest")
library(randomForest)
library(caret)

set.seed(123)
rf_base <- randomForest(charges ~ ., 
                        data  = train_set,
                        ntree = 500,
                        importance = TRUE)

# Resumen del modelo
print(rf_base)

# Importancia de variables
importance_df <- as.data.frame(importance(rf_base))
importance_df <- importance_df[order(importance_df$`%IncMSE`, 
                                     decreasing = TRUE), ]
print(round(importance_df, 2))   # sin la columna Variable, no hay conflicto

# Métricas en test
pred_rf_base <- predict(rf_base, newdata = test_set)

mae_rf_base  <- mean(abs(pred_rf_base - test_set$charges))
rmse_rf_base <- sqrt(mean((pred_rf_base - test_set$charges)^2))
r2_rf_base   <- cor(pred_rf_base, test_set$charges)^2

cat("\n--- Random Forest base ---\n")
cat("MAE: ", round(mae_rf_base,  2), "\n")
cat("RMSE:", round(rmse_rf_base, 2), "\n")
cat("R²:  ", round(r2_rf_base,   4), "\n")

#======= Ajuste con caret =======#
grid_rf <- expand.grid(mtry = c(2, 3, 4, 5, 6, 7, 8))
ctrl_rf  <- trainControl(method = "cv", number = 10)

set.seed(123)
rf_tuned <- train(charges ~ .,
                  data      = train_set,
                  method    = "rf",
                  trControl = ctrl_rf,
                  tuneGrid  = grid_rf,
                  ntree     = 500,
                  metric    = "RMSE")

cat("\nMtry óptimo:", rf_tuned$bestTune$mtry, "\n")
plot(rf_tuned, main = "Random Forest — Selección de mtry")

#====== Métricas del modelo ajustado =======
pred_rf  <- predict(rf_tuned, newdata = test_set)

mae_rf   <- mean(abs(pred_rf - test_set$charges))
rmse_rf  <- sqrt(mean((pred_rf - test_set$charges)^2))
r2_rf    <- cor(pred_rf, test_set$charges)^2

cat("\n--- Random Forest ajustado ---\n")
cat("MAE: ", round(mae_rf,  2), "\n")
cat("RMSE:", round(rmse_rf, 2), "\n")
cat("R²:  ", round(r2_rf,   4), "\n")

cat("\n===== COMPARACIÓN ACUMULADA =====\n")
cat(sprintf("%-30s MAE=%8.2f  RMSE=%8.2f  R²=%.4f\n",
            "Lineal original",      mae_lineal,    rmse_lineal,    r2_lineal))
cat(sprintf("%-30s MAE=%8.2f  RMSE=%8.2f  R²=%.4f\n",
            "Lineal v2 + features", mae_lineal_v2, rmse_lineal_v2, r2_lineal_v2))
cat(sprintf("%-30s MAE=%8.2f  RMSE=%8.2f  R²=%.4f\n",
            "Ridge",                mae_ridge,     rmse_ridge,     r2_ridge))
cat(sprintf("%-30s MAE=%8.2f  RMSE=%8.2f  R²=%.4f\n",
            "Lasso",                mae_lasso,     rmse_lasso,     r2_lasso))
cat(sprintf("%-30s MAE=%8.2f  RMSE=%8.2f  R²=%.4f\n",
            "ALasso corregido",     mae_alasso_s,  rmse_alasso_s,  r2_alasso_s))
cat(sprintf("%-30s MAE=%8.2f  RMSE=%8.2f  R²=%.4f\n",
            "RF base",              mae_rf_base,   rmse_rf_base,   r2_rf_base))
cat(sprintf("%-30s MAE=%8.2f  RMSE=%8.2f  R²=%.4f\n",
            "RF ajustado",          mae_rf,        rmse_rf,        r2_rf))

#======== XG Boost ======== #
install.packages("xgboost")
library(xgboost)
library(Matrix)

# XGBoost requiere matrices dispersas, no data.frames
X_train_xgb <- sparse.model.matrix(charges ~ . - 1, data = train_set)
X_test_xgb  <- sparse.model.matrix(charges ~ . - 1, data = test_set)
y_train_xgb <- train_set$charges
y_test_xgb  <- test_set$charges

# Verificar
dim(X_train_xgb)  # 937 x 13
dim(X_test_xgb)   # 400 x 13

# DMatrix es el formato optimizado de XGBoost
dtrain <- xgb.DMatrix(data  = X_train_xgb, 
                      label = y_train_xgb)

dtest  <- xgb.DMatrix(data  = X_test_xgb,  
                      label = y_test_xgb)

params_base <- list(
  objective        = "reg:squarederror",
  max_depth        = 4,
  eta              = 0.1,
  subsample        = 0.8,
  colsample_bytree = 0.8,
  verbosity        = 0
)

set.seed(123)
xgb_base <- xgb.train(
  params  = params_base,
  data    = dtrain,
  nrounds = 200
)

pred_xgb_base  <- predict(xgb_base, dtest)
mae_xgb_base   <- mean(abs(pred_xgb_base - y_test_xgb))
rmse_xgb_base  <- sqrt(mean((pred_xgb_base - y_test_xgb)^2))
r2_xgb_base    <- cor(pred_xgb_base, y_test_xgb)^2

cat("\n--- XGBoost base (corregido) ---\n")
cat("MAE: ", round(mae_xgb_base,  2), "\n")
cat("RMSE:", round(rmse_xgb_base, 2), "\n")
cat("R²:  ", round(r2_xgb_base,   4), "\n")

# Grid search manual sobre max_depth y eta
grid_manual <- expand.grid(
  max_depth = c(3, 4, 5, 6),
  eta       = c(0.05, 0.1, 0.15, 0.2)
)

resultados_grid <- data.frame()

for(i in 1:nrow(grid_manual)) {
  
  params_i <- list(
    objective        = "reg:squarederror",
    max_depth        = grid_manual$max_depth[i],
    eta              = grid_manual$eta[i],
    subsample        = 0.8,
    colsample_bytree = 0.8,
    verbosity        = 0
  )
  
  set.seed(123)
  cv_i <- xgb.cv(
    params                = params_i,
    data                  = dtrain,
    nrounds               = 400,
    nfold                 = 10,
    verbose               = FALSE,
    early_stopping_rounds = 20,
    maximize              = FALSE
  )
  
  # nrounds óptimo según CV
  mejor_n   <- which.min(cv_i$evaluation_log$test_rmse_mean)
  mejor_rmse <- round(min(cv_i$evaluation_log$test_rmse_mean), 2)
  
  resultados_grid <- rbind(resultados_grid, data.frame(
    max_depth    = grid_manual$max_depth[i],
    eta          = grid_manual$eta[i],
    best_nrounds = mejor_n,
    rmse_cv      = mejor_rmse
  ))
  
  # Ver progreso mientras corre
  cat(sprintf("max_depth=%d  eta=%.2f  nrounds=%d  RMSE=%.2f\n",
              grid_manual$max_depth[i], grid_manual$eta[i], 
              mejor_n, mejor_rmse))
}

# Resultados ordenados
cat("\n===== GRID SEARCH — mejores a peores =====\n")
print(resultados_grid[order(resultados_grid$rmse_cv), ])

# Tomar la primera fila (mejor RMSE)
mejor_fila <- resultados_grid[which.min(resultados_grid$rmse_cv), ]
cat("\nMejores parámetros:\n")
print(mejor_fila)

params_final <- list(
  objective        = "reg:squarederror",
  max_depth        = mejor_fila$max_depth,
  eta              = mejor_fila$eta,
  subsample        = 0.8,
  colsample_bytree = 0.8,
  verbosity        = 0
)

set.seed(123)
xgb_final <- xgb.train(
  params  = params_final,
  data    = dtrain,
  nrounds = mejor_fila$best_nrounds
)

pred_xgb  <- predict(xgb_final, dtest)
mae_xgb   <- mean(abs(pred_xgb - y_test_xgb))
rmse_xgb  <- sqrt(mean((pred_xgb - y_test_xgb)^2))
r2_xgb    <- cor(pred_xgb, y_test_xgb)^2

cat("\n--- XGBoost ajustado ---\n")
cat("MAE: ", round(mae_xgb,  2), "\n")
cat("RMSE:", round(rmse_xgb, 2), "\n")
cat("R²:  ", round(r2_xgb,   4), "\n")

# Tabla final
cat("\n===== COMPARACIÓN FINAL =====\n")
cat(sprintf("%-30s MAE=%8.2f  RMSE=%8.2f  R²=%.4f\n",
            "Lineal original",      mae_lineal,    rmse_lineal,    r2_lineal))
cat(sprintf("%-30s MAE=%8.2f  RMSE=%8.2f  R²=%.4f\n",
            "Lineal v2 + features", mae_lineal_v2, rmse_lineal_v2, r2_lineal_v2))
cat(sprintf("%-30s MAE=%8.2f  RMSE=%8.2f  R²=%.4f\n",
            "Ridge",                mae_ridge,     rmse_ridge,     r2_ridge))
cat(sprintf("%-30s MAE=%8.2f  RMSE=%8.2f  R²=%.4f\n",
            "Lasso",                mae_lasso,     rmse_lasso,     r2_lasso))
cat(sprintf("%-30s MAE=%8.2f  RMSE=%8.2f  R²=%.4f\n",
            "ALasso corregido",     mae_alasso_s,  rmse_alasso_s,  r2_alasso_s))
cat(sprintf("%-30s MAE=%8.2f  RMSE=%8.2f  R²=%.4f\n",
            "RF base",              mae_rf_base,   rmse_rf_base,   r2_rf_base))
cat(sprintf("%-30s MAE=%8.2f  RMSE=%8.2f  R²=%.4f\n",
            "RF ajustado",          mae_rf,        rmse_rf,        r2_rf))
cat(sprintf("%-30s MAE=%8.2f  RMSE=%8.2f  R²=%.4f\n",
            "XGBoost base",         mae_xgb_base,  rmse_xgb_base,  r2_xgb_base))
cat(sprintf("%-30s MAE=%8.2f  RMSE=%8.2f  R²=%.4f\n",
            "XGBoost ajustado",     mae_xgb,       rmse_xgb,       r2_xgb))

# =======STACKING =======

# ======= caretEnsemble =======


library(caret)
library(caretEnsemble)
library(ggplot2)

ctrl_stack <- trainControl(
  method          = "cv",
  number          = 10,
  savePredictions = "final",  
  allowParallel   = FALSE
)

# ======= Grids de hiperparámetros =======
tune_rf <- expand.grid(mtry = c(2, 3, 4, 5, 6))

tune_glmnet <- expand.grid(
  alpha  = c(0, 0.25, 0.5, 0.75, 1),
  lambda = 10^seq(-4, 2, length = 20)
)

# ======= Entrenamiento de los modelos base ======= 
set.seed(123)
model_list <- caretList(
  charges ~ .,
  data      = train_set,
  trControl = ctrl_stack,
  tuneList  = list(
    lm     = caretModelSpec(method = "lm"),
    rf     = caretModelSpec(method = "rf",     tuneGrid = tune_rf),
    glmnet = caretModelSpec(method = "glmnet", tuneGrid = tune_glmnet)
  )
)

cat("\n--- Métricas individuales de cada modelo base (CV) ---\n")
res <- resamples(model_list)
print(summary(res))

cat("\n--- Correlación entre predicciones (diversidad) ---\n")
# Correlación baja = mayor beneficio al apilar
print(round(modelCor(res), 3))

cat("\n>>> Entrenando meta-modelo glmnet...\n")

set.seed(123)
stack_glmnet <- caretStack(
  model_list,
  method    = "glmnet",
  trControl = trainControl(method = "cv", number = 10),
  tuneGrid  = expand.grid(
    alpha  = c(0, 0.5, 1),
    lambda = 10^seq(-4, 2, length = 30)
  )
)

cat("\nMejor lambda del meta-modelo glmnet:\n")
print(stack_glmnet$ens_model$bestTune)

cat("\n>>> Entrenando meta-modelo rf...\n")

set.seed(123)
stack_rf <- caretStack(
  model_list,
  method    = "rf",
  trControl = trainControl(method = "cv", number = 10),
  tuneGrid  = expand.grid(mtry = c(1, 2, 3))
)

eval_model <- function(pred, actual, label) {
  mae  <- mean(abs(pred - actual))
  rmse <- sqrt(mean((pred - actual)^2))
  r2   <- cor(pred, actual)^2
  cat(sprintf("%-38s MAE=%8.2f  RMSE=%8.2f  R²=%.4f\n",
              label, mae, rmse, r2))
  invisible(list(mae = mae, rmse = rmse, r2 = r2))
}

pred_stack_glmnet <- predict(stack_glmnet, newdata = test_set)$pred
pred_stack_rf     <- predict(stack_rf,     newdata = test_set)$pred

cat("\n===== RESULTADOS STACKING (caretEnsemble) =====\n")
m_sg <- eval_model(pred_stack_glmnet, test_set$charges, "Stack meta=glmnet")
m_sr <- eval_model(pred_stack_rf,     test_set$charges, "Stack meta=rf")

# Seleccionar el mejor para la tabla final
if (m_sg$r2 >= m_sr$r2) {
  best_caret_pred  <- pred_stack_glmnet
  best_caret_label <- "Stack caret (meta=glmnet)"
  best_caret_m     <- m_sg
} else {
  best_caret_pred  <- pred_stack_rf
  best_caret_label <- "Stack caret (meta=rf)"
  best_caret_m     <- m_sr
}


ggplot(data.frame(real = test_set$charges, predicho = best_caret_pred),
       aes(x = real, y = predicho)) +
  geom_point(alpha = 0.4, color = "#5B5EA6") +
  geom_abline(slope = 1, intercept = 0,
              color = "red", linetype = "dashed", linewidth = 1) +
  annotate("text", x = 5000, y = 55000,
           label = paste0("R² = ", round(best_caret_m$r2, 4)),
           size = 5, color = "#0073C2FF", fontface = "bold") +
  labs(title    = paste(best_caret_label, "— Real vs Predicho"),
       subtitle = "Línea roja = predicción perfecta",
       x = "Charges real (USD)", y = "Charges predicho (USD)") +
  theme_bw()

#  ======= H2O AutoML =======

options(timeout = 600)

if (!requireNamespace("h2o", quietly = TRUE))
  install.packages("h2o")

library(h2o)

h2o.init(nthreads = -1, max_mem_size = "4G")
h2o.no_progress()

# ── Convertir a formato H2O =======
train_h2o <- as.h2o(train_set)
test_h2o  <- as.h2o(test_set)

y_col  <- "charges"
x_cols <- setdiff(names(train_set), y_col)

cat("\nVariables predictoras (H2O):\n")
print(x_cols)

cat("\n>>> Iniciando H2O AutoML (máx 5 min o 30 modelos)...\n")

aml <- h2o.automl(
  x                                = x_cols,
  y                                = y_col,
  training_frame                   = train_h2o,
  max_models                       = 30,
  max_runtime_secs                 = 3000,
  seed                             = 123,
  sort_metric                      = "RMSE",
  include_algos                    = c("GLM", "DRF", "GBM",
                                       "XGBoost", "DeepLearning",
                                       "StackedEnsemble"),
  keep_cross_validation_predictions = TRUE,
  exploitation_ratio               = 0.1
)

lb    <- h2o.get_leaderboard(aml, extra_columns = "ALL")
lb_df <- as.data.frame(lb)

cat("\n--- H2O AutoML Leaderboard (top 10) ---\n")
print(head(lb_df[, c("model_id", "rmse", "mae", "mean_residual_deviance")], 10))

# ======= Mejor modelo global =======
best_h2o    <- aml@leader
pred_leader <- as.vector(h2o.predict(best_h2o, test_h2o))
m_leader    <- eval_model(pred_leader, test_set$charges,
                          paste0("H2O leader (", best_h2o@algorithm, ")"))

se_ids <- lb_df$model_id[grepl("StackedEnsemble", lb_df$model_id)]

if (length(se_ids) > 0) {
  best_se  <- h2o.getModel(se_ids[1])
  pred_se  <- as.vector(h2o.predict(best_se, test_h2o))
  m_se     <- eval_model(pred_se, test_set$charges,
                         "H2O StackedEnsemble (mejor)")
} else {
  cat("  (No se encontró StackedEnsemble en el leaderboard)\n")
  m_se     <- m_leader
  pred_se  <- pred_leader
}

pred_h2o_plot <- if (length(se_ids) > 0) pred_se  else pred_leader
r2_h2o_plot   <- if (length(se_ids) > 0) m_se$r2  else m_leader$r2

ggplot(data.frame(real = test_set$charges, predicho = pred_h2o_plot),
       aes(x = real, y = predicho)) +
  geom_point(alpha = 0.4, color = "#20854E") +
  geom_abline(slope = 1, intercept = 0,
              color = "red", linetype = "dashed", linewidth = 1) +
  annotate("text", x = 5000, y = 55000,
           label = paste0("R² = ", round(r2_h2o_plot, 4)),
           size = 5, color = "#0073C2FF", fontface = "bold") +
  labs(title    = "H2O AutoML — Real vs Predicho",
       subtitle = "Línea roja = predicción perfecta",
       x = "Charges real (USD)", y = "Charges predicho (USD)") +
  theme_bw()


# ======= Comparación acumulada completa =======

cat("\n\n===== COMPARACIÓN ACUMULADA COMPLETA =====\n")
cat(sprintf("%-38s %s\n", "Modelo", "     MAE        RMSE       R²"))
cat(strrep("-", 76), "\n")

cat(sprintf("%-38s MAE=%8.2f  RMSE=%8.2f  R²=%.4f\n",
            "Lineal original",      mae_lineal,    rmse_lineal,    r2_lineal))
cat(sprintf("%-38s MAE=%8.2f  RMSE=%8.2f  R²=%.4f\n",
            "Lineal v2 + features", mae_lineal_v2, rmse_lineal_v2, r2_lineal_v2))
cat(sprintf("%-38s MAE=%8.2f  RMSE=%8.2f  R²=%.4f\n",
            "Ridge",                mae_ridge,     rmse_ridge,     r2_ridge))
cat(sprintf("%-38s MAE=%8.2f  RMSE=%8.2f  R²=%.4f\n",
            "Lasso",                mae_lasso,     rmse_lasso,     r2_lasso))
cat(sprintf("%-38s MAE=%8.2f  RMSE=%8.2f  R²=%.4f\n",
            "ALasso corregido",     mae_alasso_s,  rmse_alasso_s,  r2_alasso_s))
cat(sprintf("%-38s MAE=%8.2f  RMSE=%8.2f  R²=%.4f\n",
            "RF base",              mae_rf_base,   rmse_rf_base,   r2_rf_base))
cat(sprintf("%-38s MAE=%8.2f  RMSE=%8.2f  R²=%.4f\n",
            "RF ajustado",          mae_rf,        rmse_rf,        r2_rf))
cat(sprintf("%-38s MAE=%8.2f  RMSE=%8.2f  R²=%.4f\n",
            "XGBoost base",         mae_xgb_base,  rmse_xgb_base,  r2_xgb_base))
cat(sprintf("%-38s MAE=%8.2f  RMSE=%8.2f  R²=%.4f\n",
            "XGBoost ajustado",     mae_xgb,       rmse_xgb,       r2_xgb))
cat(strrep("-", 76), "\n")
cat(sprintf("%-38s MAE=%8.2f  RMSE=%8.2f  R²=%.4f\n",
            "Stack caret (meta=glmnet)", m_sg$mae, m_sg$rmse, m_sg$r2))
cat(sprintf("%-38s MAE=%8.2f  RMSE=%8.2f  R²=%.4f\n",
            "Stack caret (meta=rf)",     m_sr$mae, m_sr$rmse, m_sr$r2))
cat(sprintf("%-38s MAE=%8.2f  RMSE=%8.2f  R²=%.4f\n",
            "H2O StackedEnsemble",       m_se$mae, m_se$rmse, m_se$r2))
cat(sprintf("%-38s MAE=%8.2f  RMSE=%8.2f  R²=%.4f\n",
            paste0("H2O leader (", best_h2o@algorithm, ")"),
            m_leader$mae, m_leader$rmse, m_leader$r2))
cat(strrep("-", 76), "\n")

# =======RMSE comparativo de todos los modelos =======
modelos_df <- data.frame(
  Modelo = c(
    "Lineal original", "Lineal v2+feat", "Ridge", "Lasso", "ALasso",
    "RF base", "RF ajust.", "XGB base", "XGB ajust.",
    "Stack caret", "H2O SE"
  ),
  RMSE = c(
    rmse_lineal, rmse_lineal_v2, rmse_ridge, rmse_lasso, rmse_alasso_s,
    rmse_rf_base, rmse_rf, rmse_xgb_base, rmse_xgb,
    best_caret_m$rmse, m_se$rmse
  ),
  Tipo = c(
    "Lineal", "Lineal", "Penalizado", "Penalizado", "Penalizado",
    "Ensamble", "Ensamble", "Ensamble", "Ensamble",
    "Stacking", "Stacking"
  )
)

modelos_df$Modelo <- factor(
  modelos_df$Modelo,
  levels = modelos_df$Modelo[order(modelos_df$RMSE, decreasing = TRUE)]
)

ggplot(modelos_df, aes(x = Modelo, y = RMSE, fill = Tipo)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = round(RMSE, 0)), hjust = -0.1, size = 3.2) +
  coord_flip() +
  scale_fill_manual(values = c(
    "Lineal"     = "#0073C2FF",
    "Penalizado" = "#EFC000FF",
    "Ensamble"   = "#CD534CFF",
    "Stacking"   = "#20854EFF"
  )) +
  labs(
    title    = "Comparación de modelos — RMSE (menor es mejor)",
    subtitle = "Dataset: Medical Cost Personal — insurance.csv",
    x = "", y = "RMSE (USD)", fill = "Familia"
  ) +
  theme_bw() +
  theme(plot.title = element_text(face = "bold"),
        legend.position = "bottom")

#=======Evolución del R² incluyendo stacking =======
evolucion_df <- data.frame(
  Paso  = 1:11,
  Modelo = c(
    "Lineal\noriginal", "Lineal v2\n+features", "Ridge",
    "Lasso", "ALasso", "RF base", "RF ajust.",
    "XGB base", "XGB ajust.", "Stack\ncaret", "H2O SE"
  ),
  R2 = c(
    r2_lineal, r2_lineal_v2, r2_ridge, r2_lasso, r2_alasso_s,
    r2_rf_base, r2_rf, r2_xgb_base, r2_xgb,
    best_caret_m$r2, m_se$r2
  )
)

ggplot(evolucion_df, aes(x = Paso, y = R2)) +
  geom_line(color = "#868686FF", linewidth = 1) +
  geom_point(aes(color = R2), size = 4) +
  geom_text(aes(label = round(R2, 4)), vjust = -0.9, size = 3) +
  scale_x_continuous(breaks = 1:11, labels = evolucion_df$Modelo) +
  scale_color_gradient(low = "#CD534CFF", high = "#0073C2FF") +
  labs(
    title = "Evolución del R² a través de todos los modelos",
    x = "", y = "R²"
  ) +
  theme_bw() +
  theme(
    axis.text.x     = element_text(size = 7),
    legend.position = "none",
    plot.title      = element_text(face = "bold")
  )

h2o.shutdown(prompt = FALSE)
cat("\nH2O apagado. Sección de stacking completada.\n")

write.csv(insurance_data,
          "/Users/alejandrojuarezrojas/Downloads/ML/insurance_final.csv",
          row.names = FALSE)
