
# Configurando o diretório com a base de dados
setwd("C:/R/Bases/Projeto01")

# Carregando pacotes
library(ggplot2)
library(caret)
library(plyr)
library(dplyr)
library(data.table)
library(randomForest)

# Coletando dados
ad <- read.csv("train_sample.csv", header = TRUE, sep = ",", na.strings =c(""))

## Variáveis
# ip: ip address of click.
# app: app id for marketing.
# device: device type id of user mobile phone (e.g., iphone 6 plus, iphone 7, huawei mate 7, etc.)
# os: os version id of user mobile phone 
# channel: channel id of mobile ad publisher
# click_time: timestamp of click (UTC)
# attributed_time: if user download the app for after clicking an ad, this is the time of the app download
# is_attributed: the target that is to be predicted, indicating the app was downloaded


## Etapa 01 - Análise Exploratória

# Ver a tabela
View(ad)

# Ver resumo dos dados
summary(ad)
str(ad)

# Seleção de varíaveis

ad$is_attributed <- as.factor(mapvalues(ad$is_attributed, from=c("0","1"), to=c("No", "Yes")))

colunas_num <- sapply(ad, is.numeric)
data_cor <- cor(ad[, colunas_num])
corrplot(data_cor,method = 'number')


var <- randomForest(is_attributed ~ ip + app + device + os + channel, data = ad, ntree = 100, nodesize = 10, importance = TRUE)
varImpPlot(var, sort=T, main = 'Variáveis Mais Importantes')


## Etapa 02 - Preparação dos Dados

# Deleta colunas de data - Optou-se por análise somente com as colunas numéricas

ad['click_time'] <- NULL
ad['attributed_time'] <- NULL

# Deleta coluna 'device' - Alta correlação com a coluna 'os'

ad['device'] <- NULL

# Separa dados de TREINO e TESTE

separadados <- createDataPartition(ad$is_attributed, p = 0.7,list=FALSE)
treino <- ad[separadados,]
teste <- ad[-separadados,]
str(treino)
str(teste)


## Etapa 03 - Testa modelos

# Regressão logística

LogModel <- glm(is_attributed ~ ., family=binomial(link="logit"), data=treino)
print(summary(LogModel))

prevlogmodel <- predict(LogModel, newdata = teste, type='response')


teste$is_attributed <- as.character(teste$is_attributed)
teste$is_attributed[teste$is_attributed=="No"] <- 0
teste$is_attributed[teste$is_attributed=="Yes"] <- 1
fitted.results <- predict(LogModel,newdata=teste,type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)
misClasificError <- mean(fitted.results != teste$is_attributed)
print(paste('Acurácia do Modelo',1-misClasificError))
print("Confusion Matrix Para Logistic Regression"); table(teste$is_attributed, fitted.results > 0.5)


#Random Florest

rfModel <- randomForest(is_attributed ~ ., data = treino)
print(rfModel)
plot(rfModel)

prevrfmodel <- predict(rfModel, newdata = teste)
print(prevrfmodel)

table(teste$is_attributed,prevrfmodel)
mean(prevrfmodel != teste$is_attributed)




### DUMP ###

library(Amelia)

train <- fread("train_sample.csv", showProgress=F)

# Verifica valores nulos
sum(is.na(ad))

missmap(ad, 
        main = "Dados Ausentes", 
        col = c("yellow", "black"), 
        legend = FALSE)

# Verifica variaveis
ggplot(aes(c)) + geom_bar()
ggplot(ad,aes(is_attributed)) + geom_bar()
ggplot(ad,aes(ip)) + geom_bar()
ggplot(as,aes(app)) + geom_bar(aes(fill = factor(Sex)), alpha = 0.5)
ggplot(ad,aes(device)) + geom_histogram(fill = 'blue', bins = 20, alpha = 0.5)
ggplot(ad,aes(os)) + geom_bar(fill = 'red', alpha = 0.5)
ggplot(ad,aes(channel)) + geom_histogram(fill = 'green', color = 'black', alpha = 0.5)

fea <- c("os", "channel", "device", "app", "attributed_time", "click_time", "ip")
train[, lapply(.SD, uniqueN), .SDcols = fea] %>%
  melt(variable.name = "features", value.name = "unique_values") %>%
  ggplot(aes(reorder(features, -unique_values), unique_values)) +
  geom_bar(stat = "identity", fill = "steelblue") + 
  scale_y_log10(breaks = c(50,100,250, 500, 10000, 50000)) +
  geom_text(aes(label = unique_values), vjust = 1.6, color = "white", size=3.5) +
  theme_minimal() +
  labs(x = "features", y = "Number of unique values")

# Calcula diferença de tempo de click e download
ad["click_time"] <- as.POSIXct(ad$click_time)
ad["attributed_time"] <- as.POSIXct(ad$attributed_time)
ad["timediff"] <- difftime(as.POSIXct(ad$click_time), as.POSIXct(ad$attributed_time), units = 'mins')
