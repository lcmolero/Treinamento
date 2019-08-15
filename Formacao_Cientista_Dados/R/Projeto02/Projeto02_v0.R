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
plot(cli)

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
dm['Cliente_ID_bl'] <- NULL
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

modelo_v2 <- lm(Demanda_uni_equil ~ Venta_uni_hoy + Dev_uni_proxima, data = treino)
summary(modelo_v2)

modelo_v3 <- lm(Demanda_uni_equil ~ Semana + Venta_uni_hoy + Dev_uni_proxima, data = treino)
summary(modelo_v3)

modelo_v4 <- lm(Demanda_uni_equil ~ Semana + Canal_ID + Venta_uni_hoy + Dev_uni_proxima, data = treino)
summary(modelo_v4)

modelo_v5 <- lm(Demanda_uni_equil ~ Semana + Canal_ID + Cliente_ID + Producto_ID + Venta_uni_hoy + Dev_uni_proxima, data = treino)
summary(modelo_v5)

varImp(modelo_v1)
varImp(modelo_v2)
varImp(modelo_v3)
varImp(modelo_v4)
varImp(modelo_v5)


## Testa Modelo

previsao_v1 <- predict(modelo_v1, teste)



# Obs 01: Por questão de sesempenho foi usado somente os dados de amostra de treino (train_sample.csv) disponível no Kaggle
# Obs 02: Foi feita análise simplificada sem realizar balanceamento e normalização dos dados.

# Melhor opção pelo modelo de Random Florest, acurácia similar a Regressão Logística mas com mais acertos 
# de verdadeiro positivo

help(quantile)
teste <- as.list(quantile(produtos$Und,names = FALSE))
class(teste)
print(teste[2])
teste[1]



  
  




###   DUMP   ###

# Coletando amostra dos dados 
#system.time(dm <- fread('train.csv', showProgress = TRUE))
#system.time(dm_sample <- sample_n(dm, 3e5))
#write.csv(dm_sample,'train_sample.csv',row.names = FALSE)
#View(dm_sample)
#help(sample_n)

# Converte váriavel alvo
#ad$is_attributed <- as.factor(mapvalues(ad$is_attributed, from=c("0","1"), to=c("No", "Yes")))

# Correlação
#colunas_num <- sapply(dm, is.numeric)
#print(colunas_num)
#class(colunas_num)
#help(sapply)
#help(cor)
#data_cor <- cor(dm[,'Semana', 'Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Venta_uni_hoy', 'Venta_hoy', 'Dev_uni_proxima','Dev_proxima'])


# 
# ggplot(produtos) + 
#   geom_bar(aes(x = factor(Producto_ID), y = Und), stat = 'identity', fill = "blue", show.legend = FALSE, alpha=0.5) +
#   xlab('Canal') +
#   ylab("Vendas Und") +
#   ggtitle("Venda de Produtos") +
#   theme_gray()

###   ANOTAÇÕES   ###
# Clusterizar primeiro por perfil/local loja?
# loja e região entram na equação?
# bloxplot e scatter?
# variáveis: Semana + Agencia_ID + Canal_ID + Ruta_SAK + Cliente_ID + NombreCliente + Producto_ID + NombreProducto + Venta_uni_hoy + Venta_hoy + Dev_uni_proxima + Dev_proxima

# lista <- c(quantile(produtos$Und, names = FALSE))
# i <- lista[1]
# j <- lista[2]
# l <- lista[3]
# m <- lista[4]
