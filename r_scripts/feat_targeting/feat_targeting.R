
# Level 1 feature targeting:

source("r_scripts/feat_targeting/feat_targeting_config.R")

this_seed <- 1776
train <- readRDS("input/train.rds")
test <- readRDS("input/test.rds")


# X1 --------------------------
# target: ps_ind_01
# type: 8 class multi
# removals: target, id, 
set.seed(this_seed)
x1_ho_indx <- caret::createDataPartition(y=train$ps_ind_01, times=1, p=0.25, list=F)
x1_train <- train[-x1_ho_indx, ]
x1_ho <- train[x1_ho_indx, ]

    # fix and rebalance
    


x1_train_y <- x1_train$ps_ind_01
x1_ho_y <- x1_ho$ps_ind_01
x1_removals <- c("ps_ind_01", "target", "id")
x1_train <- x1_train[, setdiff(names(x1_train), x1_removals)]
x1_ho <- x1_ho[, setdiff(names(x1_ho), x1_removals)]
x1_train_dmat <- xgb.DMatrix(as.matrix(x1_train), label=x1_train_y)
x1_ho_dmat <- xgb.DMatrix(as.matrix(x1_ho), label=x1_ho_y)
x1_params <-  list(
    # "objective" = "binary:logistic",
    # "eval_metric" = "auc",  
    "objective" = "multi:softprob",
    "eval_metric" = "mlogloss",
    "num_class" = length(unique(x1_train_y)),  # <-- 8 classes including "0"
    "eta" = 0.01,
    "max_depth" = 7,
    "subsample" = 0.8,
    "colsample_bytree" = 0.5,
    "lambda" = 0,
    "alpha" = 1,
    "gamma" = 0,
    "max_delta_step" = 1,
    "scale_pos_weight" = 1,
    "nthread" = 4)
x1_xgb_cv <- xgb.cv(
    data= x1_train_dmat,
    nfold=5,
    params=x1_params,
    nrounds=4000,
    early_stopping_rounds = 50,
    print_every_n = 5)
x1_nrounds <- which.min(x1_xgb_cv$evaluation_log$test_rmse_mean)
x1_xgb <- xgboost::xgb.train(
    data=x1_train_dmat,
    nrounds=x1_nrounds,
    params = x1_params)
x1_feat_imp <- xgboost::xgb.importance(feature_names = names(x1_train), model=x1_xgb)
xgboost::xgb.plot.importance(x1_feat_imp[1:20,])
x1_preds <- as.data.frame(matrix(predict(x1_xgb, x1_ho_dmat), ncol=length(unique(x1_train_y)), byrow=T))
names(x1_preds) <- paste0("feat_tar_", "ps_ind_01", "_", 0:7)
x1_preds <- cbind(x1_ho_y, x1_preds)
saveRDS(x1_xgb, file="cache/feat_targeting/ft_mods/ft01_ps_ind_01_mod.rds")
saveRDS(names(x1_train), file="cache/feat_targeting/ft_feats/ft01_ps_ind_01_feats.rds")




# x2 --------------------------
# target: ps_ind_02_cat
# type: 8 class multi
# removals: target, id, 
train$ps_ind_02_cat %>% table()
set.seed(this_seed)
x2_ho_indx <- caret::createDataPartition(y=train$ps_ind_02_cat, times=1, p=0.25, list=F)
x2_train <- train[-x2_ho_indx, ]
x2_ho <- train[x2_ho_indx, ]
    
    # fix and rebalance -- need to do this for test as well
    x2_train <- x2_train[x2_train$ps_ind_02_cat > 0, ]
    x2_ho <- x2_ho[x2_ho$ps_ind_02_cat > 0, ]
    x2_train <- rbind(
         x2_train[x2_train$ps_ind_02_cat == 1,] %>% sample_n(100000, replace=T),
         x2_train[x2_train$ps_ind_02_cat == 2,] %>% sample_n(100000, replace=T),
         x2_train[x2_train$ps_ind_02_cat == 3,] %>% sample_n(100000, replace=T),
         x2_train[x2_train$ps_ind_02_cat == 4,] %>% sample_n(100000, replace=T)
    )[sample(1:400000, 400000, replace = F), ]    
    table(x2_train$ps_ind_02_cat)

# subtracting by one to get it betwee 0 - numberofclassesminusone
x2_train_y <- (x2_train$ps_ind_02_cat - 1)
x2_ho_y <- (x2_ho$ps_ind_02_cat - 1)
x2_removals <- c("ps_ind_02_cat", "target", "id")
x2_train <- x2_train[, setdiff(names(x2_train), x2_removals)]
x2_ho <- x2_ho[, setdiff(names(x2_ho), x2_removals)]
x2_train_dmat <- xgb.DMatrix(as.matrix(x2_train), label=x2_train_y)
x2_ho_dmat <- xgb.DMatrix(as.matrix(x2_ho), label=x2_ho_y)
x2_params <-  list(
    # "objective" = "binary:logistic",
    # "eval_metric" = "auc",  
    "objective" = "multi:softprob",
    "eval_metric" = "mlogloss",
    "num_class" = length(unique(x2_train_y)),  # <-- 8 classes including "0"
    "eta" = 0.25,
    "max_depth" = 7,
    "subsample" = 0.8,
    "colsample_bytree" = 0.5,
    "lambda" = 0,
    "alpha" = 1,
    "gamma" = 0,
    "max_delta_step" = 0,
    "scale_pos_weight" = 1,
    "nthread" = 4)
x2_xgb_cv <- xgb.cv(
    data= x2_train_dmat,
    params=x2_params,
    nfold=5,
    nrounds=10000,
    early_stopping_rounds = 50,
    print_every_n = 5)
x2_nrounds <- which.min(x2_xgb_cv$evaluation_log$test_rmse_mean)
x2_xgb <- xgboost::xgb.train(
    data=x2_train_dmat,
    nrounds=x2_nrounds,
    params = x2_params)
x2_feat_imp <- xgboost::xgb.importance(feature_names = names(x2_train), model=x2_xgb)
xgboost::xgb.plot.importance(x2_feat_imp[1:20,])
x2_preds <- as.data.frame(matrix(predict(x2_xgb, x2_ho_dmat), ncol=length(unique(x2_train_y)), byrow=T))
names(x2_preds) <- paste0("feat_tar_", "ps_ind_02_cat", "_", 0:7)
x2_preds <- cbind(x2_ho_y, x2_preds)
saveRDS(x2_xgb, file="cache/feat_targeting/ft_mods/ft01_ps_ind_01_mod.R")
saveRDS(names(x2_train), file="cache/feat_targeting/ft_feats/ft01_ps_ind_01_feats.R")

