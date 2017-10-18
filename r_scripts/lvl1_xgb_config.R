

# level 1 config


library(xgboost)
library(dplyr)
library(ggplot2)
library(caret)
library(assertthat)


data_cache_fp <- "cache/level1_files.RData"
exp_level <- "01"
fp_results <- "r_scripts/level1_results.csv"
fp_dir_models <- "cache/level1_models"
fp_dir_feats <- "cache/level1_feats"
fp_dir_preds <- "cache/level1_preds"

