
## Configurações iniciais

# Configurando o diretório com a base de dados
setwd("C:/R/Bases/Projeto01")

# Carregando pacotes
library(ggplot2)
library(caret)
library(plyr)
library(dplyr)
library(data.table)
library(randomForest)
library(corrplot)
library(e1071)


## Coletando dados

#ad <- fread('train.csv', showProgress = TRUE, nrows = 3e6)

ad <- read.csv("train_sample.csv", header = TRUE, sep = ",", na.strings =c(""))

# Variáveis:
# ip: ip address of click.
# app: app id for marketing.
# device: device type id of user mobile phone (e.g., iphone 6 plus, iphone 7, huawei mate 7, etc.)
# os: os version id of user mobile phone 
# channel: channel id of mobile ad publisher
# click_time: timestamp of click (UTC)
# attributed_time: if user download the app for after clicking an ad, this is the time of the app download
# is_attributed: the target that is to be predicted, indicating the app was downloaded


## Análise Exploratória

# Ver a tabela
View(ad)

# Ver resumo dos dados
summary(ad)
str(ad)

## Seleção de varíaveis

# Converte váriavel alvo
ad$is_attributed <- as.factor(mapvalues(ad$is_attributed, from=c("0","1"), to=c("No", "Yes")))

# Verifica correlação
colunas_num <- sapply(ad, is.numeric)
data_cor <- cor(ad[, colunas_num])
corrplot(data_cor,method = 'number')

# Verifica importância das variáveis numéricas 
var <- randomForest(is_attributed ~ ip + app + device + os + channel, data = ad, ntree = 100, nodesize = 10, importance = TRUE)
varImpPlot(var, sort=T, main = 'Variáveis Mais Importantes')


## Preparação dos Dados

# Deleta colunas de data - Optou-se por análise somente com as colunas numéricas
ad['click_time'] <- NULL
ad['attributed_time'] <- NULL

# Deleta coluna 'device' - Alta correlação com a coluna 'os'
ad['device'] <- NULL
#ad['os'] <- NULL

# Separa dados de TREINO e TESTE
separadados <- createDataPartition(ad$is_attributed, p = 0.7,list=FALSE)
treino <- ad[separadados,]
teste <- ad[-separadados,]
str(treino)
str(teste)


## Testa modelos

# Random Florest
rfModel <- randomForest(is_attributed ~ ., data = treino)
print(rfModel)
plot(rfModel)

prevrfmodel <- predict(rfModel, newdata = teste)

table(teste$is_attributed,prevrfmodel)
acurf <- mean(prevrfmodel != teste$is_attributed)
print(paste('Acurácia: ', 1-acurf))

# Naive Bayes
nbModel <- naiveBayes(is_attributed ~ ., data = treino)
print(nbModel)

prevnbmodel <- predict(nbModel, newdata = teste)

table(teste$is_attributed,prevnbmodel)
acunb <- mean(prevnbmodel != teste$is_attributed)
print(paste('Acurácia: ', 1-acunb))

# Regressão logística
logModel <- glm(is_attributed ~ ., family=binomial(link="logit"), data=treino)
print(summary(logModel))

prevlogmodel <- predict(logModel, newdata = teste, type='response')

teste$is_attributed <- as.character(teste$is_attributed)
teste$is_attributed[teste$is_attributed=="No"] <- "0"
teste$is_attributed[teste$is_attributed=="Yes"] <- "1"
fitted.results <- prevlogmodel
fitted.results <- ifelse(fitted.results > 0.5,1,0)
table(teste$is_attributed, fitted.results > 0.5)
acurl <- mean(fitted.results != teste$is_attributed)
print(paste('Acurácia: ',1-acurl))


# Obs 01: Por questão de sesempenho foi usado somente os dados de amostra de treino (train_sample.csv) disponível no Kaggle
# Obs 02: Foi feita análise simplificada sem realizar balanceamento e normalização dos dados.

# Melhor opção pelo modelo de Random Florest, acurácia similar a Regressão Logística mas com mais acertos 
# de verdadeiro positivo