



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
x1_target <- "ps_ind_01"
table(train[, x1_target])
x1_ho_indx <- caret::createDataPartition(y=train[, x1_target], times=1, p=0.25, list=F)
x1_train <- train[-x1_ho_indx, ]
x1_ho <- train[x1_ho_indx, ]
x1_samp_size <- 15000
    # fix and rebalance -- we can probably use much smaller samples
    x1_train_bal <- rbind(
        x1_train %>% filter(ps_ind_01 == 0) %>% sample_n(x1_samp_size, replace=nrow(.) < x1_samp_size),
        x1_train %>% filter(ps_ind_01 == 1) %>% sample_n(x1_samp_size, replace=nrow(.) < x1_samp_size),
        x1_train %>% filter(ps_ind_01 == 2) %>% sample_n(x1_samp_size, replace=nrow(.) < x1_samp_size),
        x1_train %>% filter(ps_ind_01 == 3) %>% sample_n(x1_samp_size, replace=nrow(.) < x1_samp_size),
        x1_train %>% filter(ps_ind_01 == 4) %>% sample_n(x1_samp_size, replace=nrow(.) < x1_samp_size),
        x1_train %>% filter(ps_ind_01 == 5) %>% sample_n(x1_samp_size, replace=nrow(.) < x1_samp_size),
        x1_train %>% filter(ps_ind_01 == 6) %>% sample_n(x1_samp_size, replace=nrow(.) < x1_samp_size),
        x1_train %>% filter(ps_ind_01 == 7) %>% sample_n(x1_samp_size, replace=nrow(.) < x1_samp_size))
    table(x1_train_bal[, x1_target])
x1_train_bal_y <- x1_train_bal[, x1_target]
x1_ho_y <- x1_ho[, x1_target]
x1_removals <- c(x1_target, "target", "id")
x1_train_bal <- x1_train_bal[, setdiff(names(x1_train_bal), x1_removals)]
x1_ho <- x1_ho[, setdiff(names(x1_ho), x1_removals)]
x1_train_bal_dmat <- xgb.DMatrix(as.matrix(x1_train_bal), label=x1_train_bal_y)
x1_ho_dmat <- xgb.DMatrix(as.matrix(x1_ho), label=x1_ho_y)
x1_params <-  list(
    # "objective" = "binary:logistic",
    # "eval_metric" = "auc",  
    "objective" = "multi:softprob",
    "eval_metric" = "mlogloss",
    "num_class" = length(unique(x1_train_bal_y)),  # <-- 8 classes including "0"
    "eta" = 0.4,
    "max_depth" = 7,
    "subsample" = 0.8,
    "colsample_bytree" = 0.5,
    "lambda" = 0,
    "alpha" = 1,
    "gamma" = 0,
    "max_delta_step" = 1,
    "scale_pos_weight" = 1,
    "nthread" = 4) # 131 best nrounds
x1_xgb_cv <- xgb.cv(
    data= x1_train_bal_dmat,
    nfold=5,
    params=x1_params,
    nrounds=4000,
    early_stopping_rounds = 50,
    print_every_n = 1)
x1_nrounds <- which.min(x1_xgb_cv$evaluation_log$test_mlogloss_mean)
x1_xgb <- xgboost::xgb.train(
    data=x1_train_bal_dmat,
    nrounds=x1_nrounds,
    params = x1_params,
    print_every_n=1)
x1_feat_imp <- xgboost::xgb.importance(feature_names = names(x1_train_bal), model=x1_xgb)
xgboost::xgb.plot.importance(x1_feat_imp[1:20,])
x1_preds <- as.data.frame(matrix(predict(x1_xgb, x1_ho_dmat), ncol=length(unique(x1_train_bal_y)), byrow=T))
names(x1_preds) <- paste0("feat_tar_x1_", x1_target, "_", 0:7)
x1_preds <- cbind(x1_ho_y, x1_preds)
x1_preds_gath <- tidyr::gather(x1_preds, pred_cat, pred_val, -x1_ho_y) %>% 
    group_by(x1_ho_y, pred_cat) %>%
    summarise(mean_pred_val = mean(pred_val, na.rm=T)) %>% ungroup
ggplot(x1_preds_gath, aes(x=x1_ho_y, y=pred_cat, fill=mean_pred_val)) +
    geom_tile(color="White", size=0.1) +
    scale_fill_viridis(name="Mean Prediction") +
    coord_equal() + 
    ggtitle(paste0("Mean Predictions: ", x1_target))
saveRDS(x1_xgb, file=paste0("cache/feat_targeting/ft_mods/ft_x1_", x1_target,  "_mod.rds"))
saveRDS(names(x1_train_bal), file=paste0("cache/feat_targeting/ft_feats/ft_x1_", x1_target,  "_feats.rds"))
rm(list=ls()[grepl("^x1_", ls())])
gc()
    #' no special instructions for gathering test predictions, just pass this column through the model as is



# x2 --------------------------
# target: ps_ind_02_cat
# type: 8 class multi
# removals: target, id, ps_ind_03
set.seed(this_seed)
x2_target <- "ps_ind_02_cat"
x2_samp_size <- 20000
table(train[, x2_target])
    
    # classes should start at zero
    x2_train <- data.frame(train)
    x2_train[, x2_target] <- x2_train[, x2_target] - 1
    table(x2_train[, x2_target])
    
x2_ho_indx <- caret::createDataPartition(y=train[, x2_target], times=1, p=0.25, list=F)
x2_train <- x2_train[-x2_ho_indx, ]
x2_ho <- x2_train[x2_ho_indx, ]
    # fix and rebalance -- we can probably use much smaller samples
    x2_train_bal <- rbind(
        x2_train %>% filter(ps_ind_02_cat == 0) %>% sample_n(x2_samp_size, replace=nrow(.) < x2_samp_size),
        x2_train %>% filter(ps_ind_02_cat == 1) %>% sample_n(x2_samp_size, replace=nrow(.) < x2_samp_size),
        x2_train %>% filter(ps_ind_02_cat == 2) %>% sample_n(x2_samp_size, replace=nrow(.) < x2_samp_size),
        x2_train %>% filter(ps_ind_02_cat == 3) %>% sample_n(x2_samp_size, replace=nrow(.) < x2_samp_size))
    table(x2_train_bal[, x2_target])
x2_train_bal_y <- x2_train_bal[, x2_target]
x2_ho_y <- x2_ho[, x2_target]
x2_removals <- c(x2_target, "target", "id", "ps_ind_03")
x2_train_bal <- x2_train_bal[, setdiff(names(x2_train_bal), x2_removals)]
x2_ho <- x2_ho[, setdiff(names(x2_ho), x2_removals)]
x2_train_bal_dmat <- xgb.DMatrix(as.matrix(x2_train_bal), label=x2_train_bal_y)
x2_ho_dmat <- xgb.DMatrix(as.matrix(x2_ho), label=x2_ho_y)
x2_params <-  list(
    # "objective" = "binary:logistic",
    # "eval_metric" = "auc",  
    "objective" = "multi:softprob",
    "eval_metric" = "mlogloss",
    "num_class" = length(unique(x2_train_bal_y)),  # <-- 8 classes including "0"
    "eta" = 0.1,
    "max_depth" = 7,
    "subsample" = 0.8,
    "colsample_bytree" = 0.5,
    "lambda" = 0,
    "alpha" = 1,
    "gamma" = 0,
    "max_delta_step" = 1,
    "scale_pos_weight" = 1,
    "nthread" = 4)
x2_xgb_cv <- xgb.cv(
    data= x2_train_bal_dmat,
    nfold=5,
    params=x2_params,
    nrounds=4000,
    early_stopping_rounds = 50,
    print_every_n = 1)
x2_nrounds <- which.min(x2_xgb_cv$evaluation_log$test_mlogloss_mean)
x2_xgb <- xgboost::xgb.train(
    data=x2_train_bal_dmat,
    nrounds=x2_nrounds,
    params = x2_params,
    print_every_n=1)
x2_feat_imp <- xgboost::xgb.importance(feature_names = names(x2_train_bal), model=x2_xgb)
xgboost::xgb.plot.importance(x2_feat_imp[1:20,])
x2_preds <- as.data.frame(matrix(predict(x2_xgb, x2_ho_dmat), ncol=length(unique(x2_train_bal_y)), byrow=T))
names(x2_preds) <- paste0("feat_tar_x2_", x2_target, "_", 0:3)
x2_preds <- cbind(x2_ho_y, x2_preds)
x2_preds_gath <- tidyr::gather(x2_preds, pred_cat, pred_val, -x2_ho_y) %>% 
    group_by(x2_ho_y, pred_cat) %>%
    summarise(mean_pred_val = mean(pred_val, na.rm=T)) %>% ungroup
ggplot(x2_preds_gath, aes(x=x2_ho_y, y=pred_cat, fill=mean_pred_val)) +
    geom_tile(color="White", size=0.1) +
    scale_fill_viridis(name="Mean Prediction") +
    coord_equal() + 
    ggtitle(paste0("Mean Predictions: ", x2_target))

#0.837 is current best
#0.993 after removing potential "cheater" variables

saveRDS(x2_xgb, file=paste0("cache/feat_targeting/ft_mods/ft_x2_", x2_target,  "_mod.rds"))
saveRDS(names(x2_train_bal), file=paste0("cache/feat_targeting/ft_feats/ft_x2_", x2_target,  "_feats.rds"))
rm(list=ls()[grepl("^x2_", ls())])
gc()




# x3 --------------------------
# target: ps_ind_03
# type: 12 multi class
# removals: target, id, 
set.seed(this_seed)
x3_target <- "ps_ind_03"
x3_samp_size <- 15000
table(train[, x3_target])

x3_train <- data.frame(train)
table(x3_train[, x3_target])

x3_ho_indx <- caret::createDataPartition(y=x3_train[, x3_target], times=1, p=0.25, list=F)
x3_ho <- x3_train[x3_ho_indx, ]
x3_train <- x3_train[-x3_ho_indx, ]

    assert_that(all(sapply(x3_ho, function(x) sum(is.na(x))) == 0))

# fix and rebalance -- we can probably use much smaller samples
x3_train_bal <- rbind(
    x3_train[x3_train[, x3_target] == 0, ] %>% sample_n(x3_samp_size, replace=nrow(.) < x3_samp_size),
    x3_train[x3_train[, x3_target] == 1, ] %>% sample_n(x3_samp_size, replace=nrow(.) < x3_samp_size),
    x3_train[x3_train[, x3_target] == 2, ] %>% sample_n(x3_samp_size, replace=nrow(.) < x3_samp_size),
    x3_train[x3_train[, x3_target] == 3, ] %>% sample_n(x3_samp_size, replace=nrow(.) < x3_samp_size),
    x3_train[x3_train[, x3_target] == 4, ] %>% sample_n(x3_samp_size, replace=nrow(.) < x3_samp_size),
    x3_train[x3_train[, x3_target] == 5, ] %>% sample_n(x3_samp_size, replace=nrow(.) < x3_samp_size),
    x3_train[x3_train[, x3_target] == 6, ] %>% sample_n(x3_samp_size, replace=nrow(.) < x3_samp_size),
    x3_train[x3_train[, x3_target] == 7, ] %>% sample_n(x3_samp_size, replace=nrow(.) < x3_samp_size),
    x3_train[x3_train[, x3_target] == 8, ] %>% sample_n(x3_samp_size, replace=nrow(.) < x3_samp_size),
    x3_train[x3_train[, x3_target] == 9, ] %>% sample_n(x3_samp_size, replace=nrow(.) < x3_samp_size),
    x3_train[x3_train[, x3_target] == 10, ] %>% sample_n(x3_samp_size, replace=nrow(.) < x3_samp_size),
    x3_train[x3_train[, x3_target] == 11, ] %>% sample_n(x3_samp_size, replace=nrow(.) < x3_samp_size))
    
table(x3_train_bal[, x3_target])
x3_train_bal_y <- x3_train_bal[, x3_target]
x3_ho_y <- x3_ho[, x3_target]
x3_removals <- c(x3_target, "target", "id", "ps_ind_02_cat")  # potential "ps_ind_02_cat"
x3_train_bal <- x3_train_bal[, setdiff(names(x3_train_bal), x3_removals)]
x3_ho <- x3_ho[, setdiff(names(x3_ho), x3_removals)]
x3_train_bal_dmat <- xgb.DMatrix(as.matrix(x3_train_bal), label=x3_train_bal_y)
x3_ho_dmat <- xgb.DMatrix(as.matrix(x3_ho), label=x3_ho_y)
x3_params <-  list(
    # "objective" = "binary:logistic",
    # "eval_metric" = "auc",  
    "objective" = "multi:softprob",
    "eval_metric" = "mlogloss",
    "num_class" = length(unique(x3_train_bal_y)),  # <-- 8 classes including "0"
    "eta" = 0.25,
    "max_depth" = 7,
    "subsample" = 0.8,
    "colsample_bytree" = 0.5,
    "lambda" = 0,
    "alpha" = 1,
    "gamma" = 0,
    "max_delta_step" = 1,
    "scale_pos_weight" = 1,
    "nthread" = 4)
x3_xgb_cv <- xgb.cv(
    data= x3_train_bal_dmat,
    nfold=5,
    params=x3_params,
    nrounds=4000,
    early_stopping_rounds = 50,
    print_every_n = 1)  # 336 nrounds 1.825 mlogloss with no cheaters removed  //  393 nrounds 1.919 mlogloss with 1 cheater removed
x3_nrounds <- which.min(x3_xgb_cv$evaluation_log$test_mlogloss_mean)
x3_xgb <- xgboost::xgb.train(
    data=x3_train_bal_dmat,
    nrounds=x3_nrounds,
    params = x3_params,
    print_every_n=1)
x3_feat_imp <- xgboost::xgb.importance(feature_names = names(x3_train_bal), model=x3_xgb)
xgboost::xgb.plot.importance(x3_feat_imp[1:20,])
x3_preds <- as.data.frame(matrix(predict(x3_xgb, x3_ho_dmat), ncol=length(unique(x3_train_bal_y)), byrow=T))
names(x3_preds) <- paste0("feat_tar_x3_", x3_target, "_", sprintf("%02.0f", 0:(length(unique(x3_train_bal_y)) - 1))    )
x3_preds <- cbind(x3_ho_y, x3_preds)
x3_preds_gath <- tidyr::gather(x3_preds, pred_cat, pred_val, -x3_ho_y) %>% 
    group_by(x3_ho_y, pred_cat) %>%
    summarise(mean_pred_val = mean(pred_val, na.rm=T)) %>% ungroup
ggplot(x3_preds_gath, aes(x=x3_ho_y, y=pred_cat, fill=mean_pred_val)) +
    geom_tile(color="White", size=0.1) +
    scale_fill_viridis(name="Mean Prediction") +
    coord_equal() + 
    ggtitle(paste0("Mean Predictions: ", x3_target))

#0.837 is current best
#0.993 after removing potential "cheater" variables

hist(x3_preds$feat_tar_x3_ps_ind_03_03)
hist(x3_preds$feat_tar_x3_ps_ind_03_03[x3_preds$x3_ho_y == 3])

saveRDS(x3_xgb, file=paste0("cache/feat_targeting/ft_mods/ft_x3_", x3_target,  "_mod.rds"))
saveRDS(names(x3_train_bal), file=paste0("cache/feat_targeting/ft_feats/ft_x3_", x3_target,  "_feats.rds"))
rm(list=ls()[grepl("^x3_", ls())])



# x4 --------------------------
# target: ps_ind_04_cat
# type: binary class
# removals: target, id, 
set.seed(this_seed)
x4_target <- "ps_ind_04_cat"
x4_samp_size <- 100000
table(train[, x4_target])

    x4_train <- data.frame(train)
    table(x4_train[, x4_target])
    x4_train <- x4_train[x4_train[, x4_target] >= 0, ]
    
x4_ho_indx <- caret::createDataPartition(y=x4_train[, x4_target], times=1, p=0.25, list=F)
x4_ho <- x4_train[x4_ho_indx, ]
x4_train <- x4_train[-x4_ho_indx, ]

    assert_that(all(sapply(x4_ho, function(x) sum(is.na(x))) == 0))

# fix and rebalance -- we can probably use much smaller samples
x4_train_bal <- rbind(
    x4_train[x4_train[, x4_target] == 0, ] %>% sample_n(x4_samp_size, replace=nrow(.) < x4_samp_size),
    x4_train[x4_train[, x4_target] == 1, ] %>% sample_n(x4_samp_size, replace=nrow(.) < x4_samp_size))

table(x4_train_bal[, x4_target])
x4_train_bal_y <- x4_train_bal[, x4_target]
x4_ho_y <- x4_ho[, x4_target]
x4_removals <- c(x4_target, "target", "id")  # ps_ind_07_bin
x4_train_bal <- x4_train_bal[, setdiff(names(x4_train_bal), x4_removals)]
x4_ho <- x4_ho[, setdiff(names(x4_ho), x4_removals)]
x4_train_bal_dmat <- xgb.DMatrix(as.matrix(x4_train_bal), label=x4_train_bal_y)
x4_ho_dmat <- xgb.DMatrix(as.matrix(x4_ho), label=x4_ho_y)
x4_params <-  list(
    "objective" = "binary:logistic",
    "eval_metric" = "auc",  
    # "objective" = "multi:softprob",
    # "eval_metric" = "mlogloss",
    # "num_class" = length(unique(x4_train_bal_y)),  # <-- 8 classes including "0"
    "eta" = 0.2,
    "max_depth" = 7,
    "subsample" = 0.8,
    "colsample_bytree" = 0.5,
    "lambda" = 0,
    "alpha" = 1,
    "gamma" = 0,
    "max_delta_step" = 1,
    "scale_pos_weight" = 1,
    "nthread" = 4)
x4_xgb_cv <- xgb.cv(
    data= x4_train_bal_dmat,
    nfold=5,
    params=x4_params,
    nrounds=4000,
    early_stopping_rounds = 50,
    print_every_n = 1)  
x4_nrounds <- which.max(x4_xgb_cv$evaluation_log$test_auc_mean)
x4_xgb <- xgboost::xgb.train(
    data=x4_train_bal_dmat,
    nrounds=x4_nrounds,
    params = x4_params,
    print_every_n=1)
x4_feat_imp <- xgboost::xgb.importance(feature_names = names(x4_train_bal), model=x4_xgb)
xgboost::xgb.plot.importance(x4_feat_imp[1:20,])
x4_preds <- data.frame(actual=x4_ho_y, preds=predict(x4_xgb, x4_ho_dmat))
ggplot(x4_preds, aes(x=preds, fill=as.factor(actual))) +
    geom_histogram(alpha=0.5, position='identity')

hist(x4_preds$feat_tar_x4_ps_ind_03_03)
hist(x4_preds$feat_tar_x4_ps_ind_03_03[x4_preds$x4_ho_y == 3])


saveRDS(x4_xgb, file=paste0("cache/feat_targeting/ft_mods/ft_x4_", x4_target,  "_mod.rds"))
saveRDS(names(x4_train_bal), file=paste0("cache/feat_targeting/ft_feats/ft_x4_", x4_target,  "_feats.rds"))
rm(list=ls()[grepl("^x4_", ls())])





# x5 --------------------------
# target: ps_ind_05_cat
# type: 7 multi class after removal of -1
# removals: target, id, 
set.seed(this_seed)
x5_target <- "ps_ind_05_cat"
x5_samp_size <- 20000
table(train[, x5_target])

x5_train <- data.frame(train)
table(x5_train[, x5_target])

x5_ho_indx <- caret::createDataPartition(y=x5_train[, x5_target], times=1, p=0.25, list=F)
x5_ho <- x5_train[x5_ho_indx, ]
x5_train <- x5_train[-x5_ho_indx, ]

    assert_that(all(sapply(x5_ho, function(x) sum(is.na(x))) == 0))
    
# fix and rebalance -- we can probably use much smaller samples
table(x5_train[, x5_target])
x5_train_bal <- rbind(
    x5_train[x5_train[, x5_target] == 0, ] %>% sample_n(x5_samp_size, replace=nrow(.) < x5_samp_size),
    x5_train[x5_train[, x5_target] == 1, ] %>% sample_n(x5_samp_size, replace=nrow(.) < x5_samp_size),
    x5_train[x5_train[, x5_target] == 2, ] %>% sample_n(x5_samp_size, replace=nrow(.) < x5_samp_size),
    x5_train[x5_train[, x5_target] == 3, ] %>% sample_n(x5_samp_size, replace=nrow(.) < x5_samp_size),
    x5_train[x5_train[, x5_target] == 4, ] %>% sample_n(x5_samp_size, replace=nrow(.) < x5_samp_size),
    x5_train[x5_train[, x5_target] == 5, ] %>% sample_n(x5_samp_size, replace=nrow(.) < x5_samp_size),
    x5_train[x5_train[, x5_target] == 6, ] %>% sample_n(x5_samp_size, replace=nrow(.) < x5_samp_size))

table(x5_train_bal[, x5_target])
x5_train_bal_y <- x5_train_bal[, x5_target]
x5_ho_y <- x5_ho[, x5_target]
x5_removals <- c(x5_target, "target", "id") 
x5_train_bal <- x5_train_bal[, setdiff(names(x5_train_bal), x5_removals)]
x5_ho <- x5_ho[, setdiff(names(x5_ho), x5_removals)]
x5_train_bal_dmat <- xgb.DMatrix(as.matrix(x5_train_bal), label=x5_train_bal_y)
x5_ho_dmat <- xgb.DMatrix(as.matrix(x5_ho), label=x5_ho_y)
x5_params <-  list(
    # "objective" = "binary:logistic",
    # "eval_metric" = "auc",  
    "objective" = "multi:softprob",
    "eval_metric" = "mlogloss",
    "num_class" = length(unique(x5_train_bal_y)),  # <-- 8 classes including "0"
    "eta" = 0.25,
    "max_depth" = 7,
    "subsample" = 0.8,
    "colsample_bytree" = 0.5,
    "lambda" = 0,
    "alpha" = 1,
    "gamma" = 0,
    "max_delta_step" = 1,
    "scale_pos_weight" = 1,
    "nthread" = 4)
x5_xgb_cv <- xgb.cv(
    data= x5_train_bal_dmat,
    nfold=5,
    params=x5_params,
    nrounds=4000,
    early_stopping_rounds = 50,
    print_every_n = 1)  
x5_nrounds <- which.min(x5_xgb_cv$evaluation_log$test_mlogloss_mean)
x5_xgb <- xgboost::xgboost(
    data=x5_train_bal_dmat,
    nrounds=x5_nrounds,
    params = x5_params,
    print_every_n=1,
    save_name = NULL,
    save_period = NULL)
x5_feat_imp <- xgboost::xgb.importance(feature_names = names(x5_train_bal), model=x5_xgb)
xgboost::xgb.plot.importance(x5_feat_imp[1:20,])
x5_preds <- as.data.frame(matrix(predict(x5_xgb, x5_ho_dmat), ncol=length(unique(x5_train_bal_y)), byrow=T))
names(x5_preds) <- paste0("feat_tar_x5_", x5_target, "_", sprintf("%02.0f", 0:(length(unique(x5_train_bal_y)) - 1))    )
x5_preds <- cbind(x5_ho_y, x5_preds)
x5_preds_gath <- tidyr::gather(x5_preds, pred_cat, pred_val, -x5_ho_y) %>% 
    group_by(x5_ho_y, pred_cat) %>%
    summarise(mean_pred_val = mean(pred_val, na.rm=T)) %>% ungroup
ggplot(x5_preds_gath, aes(x=x5_ho_y, y=pred_cat, fill=mean_pred_val)) +
    geom_tile(color="White", size=0.1) +
    scale_fill_viridis(name="Mean Prediction") +
    coord_equal() + 
    ggtitle(paste0("Mean Predictions: ", x5_target))

# this model is kind of terrible, but this should be alright
saveRDS(x5_xgb, file=paste0("cache/feat_targeting/ft_mods/ft_x5_", x5_target,  "_mod.rds"))
saveRDS(names(x5_train_bal), file=paste0("cache/feat_targeting/ft_feats/ft_x5_", x5_target,  "_feats.rds"))
rm(list=ls()[grepl("^x5_", ls())])





# verry obvious cheater variables here
# x6 --------------------------
# target: ps_ind_06_bin
# type: binary class
# removals: target, id, 
set.seed(this_seed)
x6_target <- "ps_ind_06_bin"
x6_samp_size <- 200000
table(train[, x6_target])

x6_train <- data.frame(train)
table(x6_train[, x6_target])
x6_train <- x6_train[x6_train[, x6_target] >= 0, ]

x6_ho_indx <- caret::createDataPartition(y=x6_train[, x6_target], times=1, p=0.25, list=F)
x6_ho <- x6_train[x6_ho_indx, ]
x6_train <- x6_train[-x6_ho_indx, ]

assert_that(all(sapply(x6_ho, function(x) sum(is.na(x))) == 0))

# fix and rebalance -- we can probably use much smaller samples
x6_train_bal <- rbind(
    x6_train[x6_train[, x6_target] == 0, ] %>% sample_n(x6_samp_size, replace=nrow(.) < x6_samp_size),
    x6_train[x6_train[, x6_target] == 1, ] %>% sample_n(x6_samp_size, replace=nrow(.) < x6_samp_size))

table(x6_train_bal[, x6_target])
x6_train_bal_y <- x6_train_bal[, x6_target]
x6_ho_y <- x6_ho[, x6_target]
x6_removals <- c(x6_target, "target", "id", "ps_ind_09_bin", "ps_ind_08_bin", "ps_ind_07_bin")  # ps_ind_07_bin
x6_train_bal <- x6_train_bal[, setdiff(names(x6_train_bal), x6_removals)]
x6_ho <- x6_ho[, setdiff(names(x6_ho), x6_removals)]
x6_train_bal_dmat <- xgb.DMatrix(as.matrix(x6_train_bal), label=x6_train_bal_y)
x6_ho_dmat <- xgb.DMatrix(as.matrix(x6_ho), label=x6_ho_y)
x6_params <-  list(
    "objective" = "binary:logistic",
    "eval_metric" = "auc",  
    # "objective" = "multi:softprob",
    # "eval_metric" = "mlogloss",
    # "num_class" = length(unique(x6_train_bal_y)),  # <-- 8 classes including "0"
    "eta" = 1.15,  # 0.4 is way too slow
    "max_depth" = 7,
    "subsample" = 0.8,
    "colsample_bytree" = 0.5,
    "lambda" = 0,
    "alpha" = 1,
    "gamma" = 0,
    "max_delta_step" = 1,
    "scale_pos_weight" = 1,
    "nthread" = 4)
x6_xgb_cv <- xgb.cv(
    data= x6_train_bal_dmat,
    nfold=5,
    params=x6_params,
    nrounds=4000,
    early_stopping_rounds = 50,
    print_every_n = 1)  
x6_nrounds <- which.max(x6_xgb_cv$evaluation_log$test_auc_mean)
x6_xgb <- xgboost::xgboost(
    data=x6_train_bal_dmat,
    nrounds=x6_nrounds,
    params = x6_params,
    print_every_n=1,
    save_name = NULL,
    save_period = NULL)
x6_feat_imp <- xgboost::xgb.importance(feature_names = names(x6_train_bal), model=x6_xgb)
xgboost::xgb.plot.importance(x6_feat_imp[1:20,])
x6_preds <- data.frame(actual=x6_ho_y, preds=predict(x6_xgb, x6_ho_dmat))
ggplot(x6_preds, aes(x=preds, fill=as.factor(actual))) +
    geom_histogram(alpha=0.5, position='identity')


saveRDS(x6_xgb, file=paste0("cache/feat_targeting/ft_mods/ft_x6_", x6_target,  "_mod.rds"))
saveRDS(names(x6_train_bal), file=paste0("cache/feat_targeting/ft_feats/ft_x6_", x6_target,  "_feats.rds"))
rm(list=ls()[grepl("^x6_", ls())])




# x7 --------------------------
# target: ps_ind_07_bin
# type: binary class
# removals: target, id, 
set.seed(this_seed)
x7_target <- "ps_ind_07_bin"
x7_samp_size <- 100000
table(train[, x7_target])

x7_train <- data.frame(train)
table(x7_train[, x7_target])
x7_train <- x7_train[x7_train[, x7_target] >= 0, ]

x7_ho_indx <- caret::createDataPartition(y=x7_train[, x7_target], times=1, p=0.25, list=F)
x7_ho <- x7_train[x7_ho_indx, ]
x7_train <- x7_train[-x7_ho_indx, ]

assert_that(all(sapply(x7_ho, function(x) sum(is.na(x))) == 0))

# fix and rebalance -- we can probably use much smaller samples
x7_train_bal <- rbind(
    x7_train[x7_train[, x7_target] == 0, ] %>% sample_n(x7_samp_size, replace=nrow(.) < x7_samp_size),
    x7_train[x7_train[, x7_target] == 1, ] %>% sample_n(x7_samp_size, replace=nrow(.) < x7_samp_size))

table(x7_train_bal[, x7_target])
x7_train_bal_y <- x7_train_bal[, x7_target]
x7_ho_y <- x7_ho[, x7_target]
x7_removals <- c(x7_target, "target", "id", "ps_ind_06_bin", "ps_ind_08_bin", "ps_ind_09_bin")  # ps_ind_07_bin
x7_train_bal <- x7_train_bal[, setdiff(names(x7_train_bal), x7_removals)]
x7_ho <- x7_ho[, setdiff(names(x7_ho), x7_removals)]
x7_train_bal_dmat <- xgb.DMatrix(as.matrix(x7_train_bal), label=x7_train_bal_y)
x7_ho_dmat <- xgb.DMatrix(as.matrix(x7_ho), label=x7_ho_y)
x7_params <-  list(
    "objective" = "binary:logistic",
    "eval_metric" = "auc",  
    # "objective" = "multi:softprob",
    # "eval_metric" = "mlogloss",
    # "num_class" = length(unique(x7_train_bal_y)),  # <-- 8 classes including "0"
    "eta" = 0.3,  # 0.4 is way too slow
    "max_depth" = 7,
    "subsample" = 0.8,
    "colsample_bytree" = 0.5,
    "lambda" = 0,
    "alpha" = 1,
    "gamma" = 0,
    "max_delta_step" = 1,
    "scale_pos_weight" = 1,
    "nthread" = 4)
x7_xgb_cv <- xgb.cv(
    data= x7_train_bal_dmat,
    nfold=5,
    params=x7_params,
    nrounds=4000,
    early_stopping_rounds = 50,
    print_every_n = 1)  
x7_nrounds <- which.max(x7_xgb_cv$evaluation_log$test_auc_mean)
x7_xgb <- xgboost::xgboost(
    data=x7_train_bal_dmat,
    nrounds=x7_nrounds,
    params = x7_params,
    print_every_n=1,
    save_name = NULL,
    save_period = NULL)
x7_feat_imp <- xgboost::xgb.importance(feature_names = names(x7_train_bal), model=x7_xgb)
xgboost::xgb.plot.importance(x7_feat_imp[1:20,])
x7_preds <- data.frame(actual=x7_ho_y, preds=predict(x7_xgb, x7_ho_dmat))
ggplot(x7_preds, aes(x=preds, fill=as.factor(actual))) +
    geom_histogram(alpha=0.5, position='identity')


saveRDS(x7_xgb, file=paste0("cache/feat_targeting/ft_mods/ft_x7_", x7_target,  "_mod.rds"))
saveRDS(names(x7_train_bal), file=paste0("cache/feat_targeting/ft_feats/ft_x7_", x7_target,  "_feats.rds"))
rm(list=ls()[grepl("^x7_", ls())])




# x8 --------------------------
# target: ps_ind_08_bin
# type: binary class
# removals: target, id, 
set.seed(this_seed)
x8_target <- "ps_ind_08_bin"
x8_samp_size <- 80000
table(train[, x8_target])

x8_train <- data.frame(train)
table(x8_train[, x8_target])
x8_train <- x8_train[x8_train[, x8_target] >= 0, ]

x8_ho_indx <- caret::createDataPartition(y=x8_train[, x8_target], times=1, p=0.25, list=F)
x8_ho <- x8_train[x8_ho_indx, ]
x8_train <- x8_train[-x8_ho_indx, ]

assert_that(all(sapply(x8_ho, function(x) sum(is.na(x))) == 0))

# fix and rebalance -- we can probably use much smaller samples
x8_train_bal <- rbind(
    x8_train[x8_train[, x8_target] == 0, ] %>% sample_n(x8_samp_size, replace=nrow(.) < x8_samp_size),
    x8_train[x8_train[, x8_target] == 1, ] %>% sample_n(x8_samp_size, replace=nrow(.) < x8_samp_size))

table(x8_train_bal[, x8_target])
x8_train_bal_y <- x8_train_bal[, x8_target]
x8_ho_y <- x8_ho[, x8_target]
x8_removals <- c(x8_target, "target", "id", "ps_ind_06_bin", "ps_ind_07_bin", "ps_ind_09_bin")  
x8_train_bal <- x8_train_bal[, setdiff(names(x8_train_bal), x8_removals)]
x8_ho <- x8_ho[, setdiff(names(x8_ho), x8_removals)]
x8_train_bal_dmat <- xgb.DMatrix(as.matrix(x8_train_bal), label=x8_train_bal_y)
x8_ho_dmat <- xgb.DMatrix(as.matrix(x8_ho), label=x8_ho_y)
x8_params <-  list(
    "objective" = "binary:logistic",
    "eval_metric" = "auc",  
    # "objective" = "multi:softprob",
    # "eval_metric" = "mlogloss",
    # "num_class" = length(unique(x8_train_bal_y)),  # <-- 8 classes including "0"
    "eta" = 1,  # 0.4 is way too slow
    "max_depth" = 6,
    "subsample" = 0.8,
    "colsample_bytree" = 0.5,
    "lambda" = 0,
    "alpha" = 1,
    "gamma" = 0,
    "max_delta_step" = 0,
    "scale_pos_weight" = 1,
    "nthread" = 4)
x8_xgb_cv <- xgb.cv(
    data= x8_train_bal_dmat,
    nfold=5,
    params=x8_params,
    nrounds=4000,
    early_stopping_rounds = 50,
    print_every_n = 1)  
x8_nrounds <- which.max(x8_xgb_cv$evaluation_log$test_auc_mean)  # maybe just set this to ~1.5k or something?
x8_xgb <- xgboost::xgboost(
    data=x8_train_bal_dmat,
    nrounds=x8_nrounds,
    params = x8_params,
    print_every_n=1,
    save_name = NULL,
    save_period = NULL)
x8_feat_imp <- xgboost::xgb.importance(feature_names = names(x8_train_bal), model=x8_xgb)
xgboost::xgb.plot.importance(x8_feat_imp[1:20,])
x8_preds <- data.frame(actual=x8_ho_y, preds=predict(x8_xgb, x8_ho_dmat))
ggplot(x8_preds, aes(x=preds, fill=as.factor(actual))) +
    geom_density(alpha=0.5, position='identity')


saveRDS(x8_xgb, file=paste0("cache/feat_targeting/ft_mods/ft_x8_", x8_target,  "_mod.rds"))
saveRDS(names(x8_train_bal), file=paste0("cache/feat_targeting/ft_feats/ft_x8_", x8_target,  "_feats.rds"))
rm(list=ls()[grepl("^x8_", ls())])





