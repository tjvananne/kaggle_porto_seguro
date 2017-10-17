

library(xgboost)
library(dplyr)
library(ggplot2)
library(caret)


list.files('input')


train <- readRDS('input/train.rds')
test <- readRDS('input/test.rds')


#' 1) split train into trainA and trainB - this our permanent lvl1 split
#' 2) train modelA on trainA and modelB on trainB
#' 3) predict with opposite data
#' 4) concat 
#' 

set.seed(1776)
train_indexes <- caret::createDataPartition(train$target, p=0.5, list=F)
trainA <- train[train_indexes, ]
trainB <- train[-train_indexes, ]


