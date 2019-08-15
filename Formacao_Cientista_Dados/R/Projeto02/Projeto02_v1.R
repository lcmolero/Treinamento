## Projeto 02 - Prevendo Demanda de Estoque com Base em Vendas

## https://www.kaggle.com/c/grupo-bimbo-inventory-demand

## Autor: Leonardo Molero    Data: 12/08/2019


## Configurações iniciais

# Configurando o diretório com a base de dados
setwd("C:/R/Bases/Projeto02")

# Carregando pacotes
library(data.table)
library(plyr)
library(dplyr)
library(corrplot)
library(ggplot2)
library(caret)


## Coletando dados
dm <- fread('train_sample.csv', showProgress = TRUE, data.table = FALSE)

## Variáveis:
# Semana — Week number (From Thursday to Wednesday)
# Agencia_ID — Sales Depot ID
# Canal_ID — Sales Channel ID
# Ruta_SAK — Route ID (Several routes = Sales Depot)
# Cliente_ID — Client ID
# NombreCliente — Client name
# Producto_ID — Product ID
# NombreProducto — Product Name
# Venta_uni_hoy — Sales unit this week (integer)
# Venta_hoy — Sales this week (unit: pesos)
# Dev_uni_proxima — Returns unit next week (integer)
# Dev_proxima — Returns next week (unit: pesos)

## Alvo:
# Demanda_uni_equil — Adjusted Demand (integer) (This is the target you will predict)


## Análise Exploratória

# Ver a tabela
View(dm)

# Ver resumo dos dados
summary(dm)
str(dm)

# Verifica valores vazios 
sum(is.na(dm))

# Histograma x Semana
ggplot(dm)+
  geom_histogram(aes(x=Semana), fill="blue", bins = 30)+
  xlab("Semana") +
  ylab("Qtde Linhas") +
  ggtitle("Histograma") +
  scale_x_continuous(breaks = (1:9))+
  theme_gray()

# Qtde de Vendas x Semana
ggplot(dm) + 
  geom_bar(aes(x = Semana, y = Venta_uni_hoy), stat = 'sum', fill = "blue",show.legend = FALSE) +
  xlab("Semana") +
  ylab("Vendas Und") +
  ggtitle("Venda de Produtos") +
  scale_x_continuous(breaks = (1:9))+
  theme_gray()

# Venda x Canal
vendas <- as.data.frame(dm %>% group_by(Canal_ID) %>% summarise(Und = sum(Venta_uni_hoy),Valor=sum(Venta_hoy)))
print(vendas)

ggplot(vendas) + 
  geom_bar(aes(x = factor(Canal_ID), y = Und), stat = 'identity', fill = "blue", show.legend = FALSE, alpha=0.5) +
  xlab('Canal') +
  ylab("Vendas Und") +
  ggtitle("Venda de Produtos") +
  theme_gray()

ggplot(vendas) + 
  geom_bar(aes(x = factor(Canal_ID), y = (Valor/1000)), stat = 'identity', fill = "red", show.legend = FALSE, alpha=0.5) +
  xlab('Canal') +
  ylab("Pesos (Milhões)") +
  ggtitle("Venda de Produtos") +
  theme_gray()

# Devolução X Canal
dev <- as.data.frame(dm %>% group_by(Canal_ID) %>% summarise(Dev_Und = sum(Dev_uni_proxima),Dev_Prev=sum(Demanda_uni_equil)))
print(dev)

ggplot(dev) + 
  geom_bar(aes(x = factor(Canal_ID), y = Dev_Und), stat = 'identity', fill = "red", show.legend = FALSE, alpha=0.5) +
  xlab('Canal') +
  ylab("DEvolução Und") +
  ggtitle("Venda de Produtos") +
  theme_gray()

ggplot(dev) + 
  geom_bar(aes(x = factor(Canal_ID), y = Dev_Prev), stat = 'identity', fill = "purple", show.legend = FALSE, alpha=0.5) +
  xlab('Canal') +
  ylab("Previsão Devolução Und") +
  ggtitle("Venda de Produtos") +
  theme_gray()

# Produto
produtos <- as.data.frame(dm %>% group_by(Producto_ID) %>% summarise(Und = sum(Venta_uni_hoy),Valor=sum(Venta_hoy)))
print(produtos)
produtos$Producto_ID <- as.factor(produtos$Producto_ID)
summary(produtos)
str(produtos)
plot(produtos)

ggplot(produtos) +
  geom_point(aes(x=Producto_ID, y=Und), show.legend = FALSE, alpha=0.5) +
  xlab('Produto') +
  ylab("Vendas Und") +
  ggtitle("Venda de Produtos") +
  theme_gray()

# Cliente
cli <- as.data.frame(dm %>% group_by(Cliente_ID) %>% summarise(Und = sum(Venta_uni_hoy),Valor=sum(Venta_hoy)))
print(cli)
cli$Cliente_ID <- as.factor(cli$Cliente_ID)
summary(cli)
str(cli)

# Verifica correlação
colunas_cor <- c('Semana', 'Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID', 'Venta_uni_hoy', 
                 'Venta_hoy', 'Dev_uni_proxima','Dev_proxima')
data_cor <- cor(dm[, colunas_cor])
corrplot(data_cor,method = 'number')


## Seleção de varíaveis

# Descarta váriavies de valor financeiro ($ / pesos) pela alta correlação com as variavéis de quantidade
dm['Dev_proxima'] <- NULL
dm['Venta_hoy'] <- NULL

# Descarta variáveis categoricas com muitos níveis / Analise simplificada / Restrições do equipamento
dm['Ruta_SAK'] <- NULL
dm['Agencia_ID'] <- NULL
dm['Cliente_ID'] <- NULL
dm['Producto_ID'] <- NULL


## Preparação dos Dados

# Transforma variáveis categoricas em tipo fator
dm$Semana <- as.factor(dm$Semana)
dm$Canal_ID <- as.factor(dm$Canal_ID)

# Separa dados de TREINO e TESTE
separadados <- createDataPartition(dm$Demanda_uni_equil, p = 0.7,list=FALSE)
treino <- dm[separadados,]
teste <- dm[-separadados,]
str(treino)
str(teste)


## Treina modelos

# Regressão Linear
modelo_v1 <- lm(Demanda_uni_equil ~ ., data = treino)
summary(modelo_v1)
varImp(modelo_v1)

modelo_v2 <- lm(Demanda_uni_equil ~ Venta_uni_hoy + Dev_uni_proxima, data = treino)
summary(modelo_v2)
varImp(modelo_v2)

modelo_v3 <- lm(Demanda_uni_equil ~ Semana + Venta_uni_hoy + Dev_uni_proxima, data = treino)
summary(modelo_v3)
varImp(modelo_v3)

modelo_v4 <- lm(Demanda_uni_equil ~ Semana + Canal_ID + Venta_uni_hoy, data = treino)
summary(modelo_v4)
varImp(modelo_v4)

modelo_v5 <- lm(Demanda_uni_equil ~ Semana + Canal_ID, data = treino)
summary(modelo_v5)
varImp(modelo_v5)

## Testa os modelos v1, v2 e v3 que tem R-Squared similar

# Testa Modelo v1
previsao_v1 <- predict(modelo_v1, teste)
plot(teste$Demanda_uni_equil, previsao_v1)

resultados <- cbind(previsao_v1, teste$Demanda_uni_equil) 
colnames(resultados) <- c('Previsto','Real')
resultados <- as.data.frame(resultados)
min(resultados)

# Função para tratar valores negativos / Não deve haver demanda de produtos negativas
trata_zero <- function(x){
  if  (x < 0){
    return(0)
  }else{
    return(x)
  }
}

# Aplicando a função para tratar valores negativos
resultados$Previsto <- sapply(resultados$Previsto, trata_zero)
resultados$Previsto

# MSE
mse <- mean((resultados$Real - resultados$Previsto)^2)
print(mse)

# RMSE
rmse <- mse^0.5
rmse

# R-Squared
SSE = sum((resultados$Previsto - resultados$Real)^2)
SST = sum((mean(dm$Demanda_uni_equil) - resultados$Real)^2)
R2_v1 = 1 - (SSE/SST)

# Testa Modelo v2
previsao_v2 <- predict(modelo_v2, teste)
plot(teste$Demanda_uni_equil, previsao_v2)

resultados <- cbind(previsao_v2, teste$Demanda_uni_equil) 
colnames(resultados) <- c('Previsto','Real')
resultados <- as.data.frame(resultados)
min(resultados)

# Aplicando a função para tratar valores negativos
resultados$Previsto <- sapply(resultados$Previsto, trata_zero)

# MSE
mse <- mean((resultados$Real - resultados$Previsto)^2)
print(mse)

# RMSE
rmse <- mse^0.5
rmse

# R-Squared
SSE = sum((resultados$Previsto - resultados$Real)^2)
SST = sum((mean(dm$Demanda_uni_equil) - resultados$Real)^2)

R2_v2 = 1 - (SSE/SST)

# Testa Modelo V3
previsao_v3 <- predict(modelo_v3, teste)
plot(teste$Demanda_uni_equil, previsao_v3)

resultados <- cbind(previsao_v3, teste$Demanda_uni_equil) 
colnames(resultados) <- c('Previsto','Real')
resultados <- as.data.frame(resultados)
min(resultados)

# Aplicando a função para tratar valores negativos
resultados$Previsto <- sapply(resultados$Previsto, trata_zero)

# MSE
mse <- mean((resultados$Real - resultados$Previsto)^2)
print(mse)

# RMSE
rmse <- mse^0.5
rmse

# R-Squared
SSE = sum((resultados$Previsto - resultados$Real)^2)
SST = sum((mean(dm$Demanda_uni_equil) - resultados$Real)^2)

R2_v3 = 1 - (SSE/SST)


## Resultado

print('Resultado Modelo 1')
print(R2_v1)

print('Resultado Modelo 2')
print(R2_v2)

print('Resultado Modelo 3')
print(R2_v3)


# Todos os modelos de regressão linear, optando por mais ou menos variáveis, resultaram no mesmo R-Squared desde que mantidas
# as variáveis 'Venta_uni_hoy' e 'Dev_uni_proxima'.

# Mostra a alta dependência do modelo nessas duas variáveis o que pode resultar em overfitting / underfitting.
# Valores negativos tiveram de ser tratados, o que não é o ideal para este caso.

# Outras varíaveis importantes foram removidas devido a grande quantidade de níveis delas, que impossibilitaram seu tratamento
# no escopo desse projeto de estudo.


### Melhor resultado: modelo_v1 
# Apresentou um R-Squared ligeiramente superior aos modelos v2 e v3


# Obs 01: Por questão de desempenho foi usado somente os dados de amostra de treino (train_sample.csv) extraída da base do Kaggle
# Obs 02: Foi feita análise simplificada sem realizar balanceamento e normalização dos dados.