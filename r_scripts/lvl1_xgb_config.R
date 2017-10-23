

# level 1 config


library(xgboost)
library(dplyr)
library(ggplot2)
library(caret)
library(assertthat)
library(glmnet)
library(corrplot)


data_cache_fp <- "cache/level1_files.RData"
exp_level <- "01"
fp_results <- "cache/level1_results.csv"
fp_dir_models <- "cache/level1_models"
fp_dir_feats <- "cache/level1_feats"
fp_dir_preds <- "cache/level1_preds"



normalizedGini <- function(aa, pp) {
    Gini <- function(a, p) {
        if (length(a) !=  length(p)) stop("Actual and Predicted need to be equal lengths!")
        temp.df <- data.frame(actual = a, pred = p, range=c(1:length(a)))
        temp.df <- temp.df[order(-temp.df$pred, temp.df$range),]
        population.delta <- 1 / length(a)
        total.losses <- sum(a)
        null.losses <- rep(population.delta, length(a)) # Hopefully is similar to accumulatedPopulationPercentageSum
        accum.losses <- temp.df$actual / total.losses # Hopefully is similar to accumulatedLossPercentageSum
        gini.sum <- cumsum(accum.losses - null.losses) # Not sure if this is having the same effect or not
        sum(gini.sum) / length(a)
    }
    Gini(aa,pp) / Gini(aa,aa)
}

