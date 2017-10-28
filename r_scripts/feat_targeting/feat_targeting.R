
#' I need to split this up into trainA and trainB for each feature stack so I can
#' build level 2 models off of these



# Level 1 feature targeting:
source("r_scripts/feat_targeting/feat_targeting_config.R")
train <- readRDS("input/train.rds")



#  use this A/B train split for training models with these feature model results
if(!file.exists("input/trainA.rds") | !file.exists("input/trainB.rds")) {
    print("trainA/B split don't exist, writing them now...")
    set.seed(1776)  # NEVER CHANGE THIS... EVER
    A_indx <- sample(1:nrow(train), size=floor(nrow(train) / 2))
    trainA <- train[A_indx, ]
    trainB <- train[-A_indx, ]
    assert_that(length(intersect(trainA$id, trainB$id)) == 0)
    rm(A_indx)
    saveRDS(trainA, "input/trainA.rds")
    saveRDS(trainB, "input/trainB.rds")
} else {
    print("trainA/B split already exist, reading them now...")
    trainA <- readRDS("input/trainA.rds")
    trainB <- readRDS("input/trainB.rds")
}


this_seed <- 1776


# rerun x15 one more time and save it out


# X1 --------------------------
# target: ps_ind_01
# type: 8 class multi
# removals: target, id, 
set.seed(this_seed)
x1_target <- "ps_ind_01"
x1_samp_size <- 20000
table(train[, x1_target])

# classes should start at zero
x1_train <- data.frame(train)
table(x1_train[, x1_target])

x1_ho_indx <- caret::createDataPartition(y=train[, x1_target], times=1, p=0.25, list=F)
x1_train <- x1_train[-x1_ho_indx, ]
x1_ho <- x1_train[x1_ho_indx, ]

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
    "eta" = 0.25,  # 0.4 is way too slow
    "max_depth" = 6,
    "subsample" = 0.8,
    "colsample_bytree" = 0.5,
    "lambda" = 0,
    "alpha" = 1,
    "gamma" = 0,
    "max_delta_step" = 0,
    "scale_pos_weight" = 1,
    "nthread" = 4)
# x8_xgb_cv <- xgb.cv(
#     data= x8_train_bal_dmat,
#     nfold=5,
#     params=x8_params,
#     nrounds=4000,
#     early_stopping_rounds = 50,
#     print_every_n = 1)  
# x8_nrounds <- which.max(x8_xgb_cv$evaluation_log$test_auc_mean)  # maybe just set this to ~1.5k or something?
x8_nrounds <- 1000  # manually setting this to see how results look, we couldn't converge
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
    geom_histogram(alpha=0.5, position='identity')


saveRDS(x8_xgb, file=paste0("cache/feat_targeting/ft_mods/ft_x8_", x8_target,  "_mod.rds"))
saveRDS(names(x8_train_bal), file=paste0("cache/feat_targeting/ft_feats/ft_x8_", x8_target,  "_feats.rds"))
rm(list=ls()[grepl("^x8_", ls())])


















# x9 --------------------------
# target: ps_ind_09_bin
# type: binary class
# removals: target, id, 
set.seed(this_seed)
x9_target <- "ps_ind_09_bin"
x9_samp_size <- 90000
table(train[, x9_target])

x9_train <- data.frame(train)
table(x9_train[, x9_target])
x9_train <- x9_train[x9_train[, x9_target] >= 0, ]

x9_ho_indx <- caret::createDataPartition(y=x9_train[, x9_target], times=1, p=0.25, list=F)
x9_ho <- x9_train[x9_ho_indx, ]
x9_train <- x9_train[-x9_ho_indx, ]

assert_that(all(sapply(x9_ho, function(x) sum(is.na(x))) == 0))

# fix and rebalance -- we can probably use much smaller samples
x9_train_bal <- rbind(
    x9_train[x9_train[, x9_target] == 0, ] %>% sample_n(x9_samp_size, replace=nrow(.) < x9_samp_size),
    x9_train[x9_train[, x9_target] == 1, ] %>% sample_n(x9_samp_size, replace=nrow(.) < x9_samp_size))

table(x9_train_bal[, x9_target])
x9_train_bal_y <- x9_train_bal[, x9_target]
x9_ho_y <- x9_ho[, x9_target]
x9_removals <- c(x9_target, "target", "id", "ps_ind_06_bin", "ps_ind_08_bin", "ps_ind_07_bin")
x9_train_bal <- x9_train_bal[, setdiff(names(x9_train_bal), x9_removals)]
x9_ho <- x9_ho[, setdiff(names(x9_ho), x9_removals)]
x9_train_bal_dmat <- xgb.DMatrix(as.matrix(x9_train_bal), label=x9_train_bal_y)
x9_ho_dmat <- xgb.DMatrix(as.matrix(x9_ho), label=x9_ho_y)
x9_params <-  list(
    "objective" = "binary:logistic",
    "eval_metric" = "auc",  
    # "objective" = "multi:softprob",
    # "eval_metric" = "mlogloss",
    # "num_class" = length(unique(x9_train_bal_y)),  # <-- 8 classes including "0"
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
x9_xgb_cv <- xgb.cv(
    data= x9_train_bal_dmat,
    nfold=5,
    params=x9_params,
    nrounds=4000,
    early_stopping_rounds = 50,
    print_every_n = 1)  
x9_nrounds <- which.max(x9_xgb_cv$evaluation_log$test_auc_mean)   # 3430 is when it converged, but it might be too good?
x9_nrounds <- 1000
x9_xgb <- xgboost::xgboost(
    data=x9_train_bal_dmat,
    nrounds=x9_nrounds,
    params = x9_params,
    print_every_n=1,
    save_name = NULL,
    save_period = NULL)
x9_feat_imp <- xgboost::xgb.importance(feature_names = names(x9_train_bal), model=x9_xgb)
xgboost::xgb.plot.importance(x9_feat_imp[1:20,])
x9_preds <- data.frame(actual=x9_ho_y, preds=predict(x9_xgb, x9_ho_dmat))
ggplot(x9_preds, aes(x=preds, fill=as.factor(actual))) +
    geom_histogram(alpha=0.5, position='identity') +
    ggtitle(x9_nrounds)


saveRDS(x9_xgb, file=paste0("cache/feat_targeting/ft_mods/ft_x9_", x9_target,  "_mod.rds"))
saveRDS(names(x9_train_bal), file=paste0("cache/feat_targeting/ft_feats/ft_x9_", x9_target,  "_feats.rds"))
rm(list=ls()[grepl("^x9_", ls())])



# x10 --------------------------
# target: ps_ind_10_bin
# type: binary class
# removals: target, id, 
# optimizing this one feels like a waste of time
set.seed(this_seed)
x10_target <- "ps_ind_10_bin"
x10_samp_size <- 1000
table(train[, x10_target])

x10_train <- data.frame(train)
table(x10_train[, x10_target])
x10_train <- x10_train[x10_train[, x10_target] >= 0, ]

x10_ho_indx <- caret::createDataPartition(y=x10_train[, x10_target], times=1, p=0.25, list=F)
x10_ho <- x10_train[x10_ho_indx, ]
x10_train <- x10_train[-x10_ho_indx, ]

assert_that(all(sapply(x10_ho, function(x) sum(is.na(x))) == 0))

# fix and rebalance -- we can probably use much smaller samples
x10_train_bal <- rbind(
    x10_train[x10_train[, x10_target] == 0, ] %>% sample_n(10000, replace=nrow(.) < x10_samp_size),
    x10_train[x10_train[, x10_target] == 1, ] %>% sample_n(x10_samp_size, replace=nrow(.) < x10_samp_size))

table(x10_train_bal[, x10_target])
x10_train_bal_y <- x10_train_bal[, x10_target]
x10_ho_y <- x10_ho[, x10_target]
x10_removals <- c(x10_target, "target", "id", "ps_ind_14", "ps_car_03_cat", "ps_car_05_cat", "ps_ind_16_bin", "ps_ind_12_bin") #, "ps_ind_06_bin", "ps_ind_08_bin", "ps_ind_07_bin")
x10_train_bal <- x10_train_bal[, setdiff(names(x10_train_bal), x10_removals)]
x10_ho <- x10_ho[, setdiff(names(x10_ho), x10_removals)]
x10_train_bal_dmat <- xgb.DMatrix(as.matrix(x10_train_bal), label=x10_train_bal_y)
x10_ho_dmat <- xgb.DMatrix(as.matrix(x10_ho), label=x10_ho_y)
x10_params <-  list(
    "objective" = "binary:logistic",
    "eval_metric" = "logloss",  
    # "objective" = "multi:softprob",
    # "eval_metric" = "mlogloss",
    # "num_class" = length(unique(x10_train_bal_y)),  # <-- 8 classes including "0"
    "eta" = 0.2,  # 0.4 is way too slow
    "max_depth" = 3,
    "subsample" = 0.8,
    "colsample_bytree" = 0.5,
    "lambda" = 0,
    "alpha" = 1,
    "gamma" = 0,
    "max_delta_step" = 1,
    "scale_pos_weight" = 11,
    "nthread" = 4)
x10_xgb_cv <- xgb.cv(
    data= x10_train_bal_dmat,
    nfold=5,
    params=x10_params,
    nrounds=4000,
    early_stopping_rounds = 50,
    print_every_n = 1)  
# x10_nrounds <- which.max(x10_xgb_cv$evaluation_log$test_auc_mean)   # 3430 is when it converged, but it might be too good?
x10_nrounds <- which.min(x10_xgb_cv$evaluation_log$test_logloss_mean)
x10_nrounds <- 30
x10_xgb <- xgboost::xgboost(
    data=x10_train_bal_dmat,
    nrounds=x10_nrounds,
    params = x10_params,
    print_every_n=1,
    save_name = NULL,
    save_period = NULL)
x10_feat_imp <- xgboost::xgb.importance(feature_names = names(x10_train_bal), model=x10_xgb)
xgboost::xgb.plot.importance(x10_feat_imp[1:20,])
x10_preds <- data.frame(actual=x10_ho_y, preds=predict(x10_xgb, x10_ho_dmat))
ggplot(x10_preds, aes(x=preds, fill=as.factor(actual))) +
    geom_density(alpha=0.5, position='identity') +
    ggtitle(x10_nrounds)


saveRDS(x10_xgb, file=paste0("cache/feat_targeting/ft_mods/ft_x10_", x10_target,  "_mod.rds"))
saveRDS(names(x10_train_bal), file=paste0("cache/feat_targeting/ft_feats/ft_x10_", x10_target,  "_feats.rds"))
rm(list=ls()[grepl("^x10_", ls())])




# x11 --------------------------
# target: ps_ind_11_bin
# type: binary class
# removals: target, id, 
set.seed(this_seed)
x11_target <- "ps_ind_11_bin"
x11_samp_size <- 1500
table(train[, x11_target])

x11_train <- data.frame(train)
table(x11_train[, x11_target])
x11_train <- x11_train[x11_train[, x11_target] >= 0, ]

x11_ho_indx <- caret::createDataPartition(y=x11_train[, x11_target], times=1, p=0.25, list=F)
x11_ho <- x11_train[x11_ho_indx, ]
x11_train <- x11_train[-x11_ho_indx, ]

assert_that(all(sapply(x11_ho, function(x) sum(is.na(x))) == 0))

# fix and rebalance -- we can probably use much smaller samples
x11_train_bal <- rbind(
    x11_train[x11_train[, x11_target] == 0, ] %>% sample_n(15000, replace=nrow(.) < x11_samp_size),
    x11_train[x11_train[, x11_target] == 1, ] %>% sample_n(x11_samp_size, replace=nrow(.) < x11_samp_size))

table(x11_train_bal[, x11_target])
x11_train_bal_y <- x11_train_bal[, x11_target]
x11_ho_y <- x11_ho[, x11_target]
x11_removals <- c(x11_target, "target", "id", "ps_ind_14", "ps_car_03_cat", "ps_ind_12_bin") #
x11_train_bal <- x11_train_bal[, setdiff(names(x11_train_bal), x11_removals)]
x11_ho <- x11_ho[, setdiff(names(x11_ho), x11_removals)]
x11_train_bal_dmat <- xgb.DMatrix(as.matrix(x11_train_bal), label=x11_train_bal_y)
x11_ho_dmat <- xgb.DMatrix(as.matrix(x11_ho), label=x11_ho_y)
x11_params <-  list(
    "objective" = "binary:logistic",
    "eval_metric" = "logloss",  
    # "objective" = "multi:softprob",
    # "eval_metric" = "mlogloss",
    # "num_class" = length(unique(x11_train_bal_y)),  # <-- 8 classes including "0"
    "eta" = 0.2,  # 0.4 is way too slow
    "max_depth" = 3,  # experimenting with turning the max depth way down
    "subsample" = 0.8,
    "colsample_bytree" = 0.5,
    "lambda" = 0,
    "alpha" = 1,
    "gamma" = 0,
    "max_delta_step" = 1,
    "scale_pos_weight" = 30,
    "nthread" = 4)
x11_xgb_cv <- xgb.cv(
    data= x11_train_bal_dmat,
    nfold=5,
    params=x11_params,
    nrounds=4000,
    early_stopping_rounds = 50,
    print_every_n = 1)  
# x11_nrounds <- which.max(x11_xgb_cv$evaluation_log$test_auc_mean)   # 3430 is when it converged, but it might be too good?
x11_nrounds <- which.min(x11_xgb_cv$evaluation_log$test_logloss_mean)
x11_nrounds <- 15
x11_xgb <- xgboost::xgboost(
    data=x11_train_bal_dmat,
    nrounds=x11_nrounds,
    params = x11_params,
    print_every_n=1,
    save_name = NULL,
    save_period = NULL)
x11_feat_imp <- xgboost::xgb.importance(feature_names = names(x11_train_bal), model=x11_xgb)
xgboost::xgb.plot.importance(x11_feat_imp[1:20,])
x11_preds <- data.frame(actual=x11_ho_y, preds=predict(x11_xgb, x11_ho_dmat))
ggplot(x11_preds, aes(x=preds, fill=as.factor(actual))) +
    geom_density(alpha=0.5, position='identity') +
    ggtitle(x11_nrounds)


saveRDS(x11_xgb, file=paste0("cache/feat_targeting/ft_mods/ft_x11_", x11_target,  "_mod.rds"))
saveRDS(names(x11_train_bal), file=paste0("cache/feat_targeting/ft_feats/ft_x11_", x11_target,  "_feats.rds"))
rm(list=ls()[grepl("^x11_", ls())])




# x12 --------------------------
# target: ps_ind_12_bin
# type: binary class
# removals: target, id, 
set.seed(this_seed)
x12_target <- "ps_ind_12_bin"
x12_samp_size <- 7000
table(train[, x12_target])

x12_train <- data.frame(train)
table(x12_train[, x12_target])
x12_train <- x12_train[x12_train[, x12_target] >= 0, ]

x12_ho_indx <- caret::createDataPartition(y=x12_train[, x12_target], times=1, p=0.25, list=F)
x12_ho <- x12_train[x12_ho_indx, ]
x12_train <- x12_train[-x12_ho_indx, ]

assert_that(all(sapply(x12_ho, function(x) sum(is.na(x))) == 0))

# fix and rebalance -- we can probably use much smaller samples
x12_train_bal <- rbind(
    x12_train[x12_train[, x12_target] == 0, ] %>% sample_n(x12_samp_size*10, replace=nrow(.) < x12_samp_size),
    x12_train[x12_train[, x12_target] == 1, ] %>% sample_n(x12_samp_size, replace=nrow(.) < x12_samp_size))

table(x12_train_bal[, x12_target])
x12_train_bal_y <- x12_train_bal[, x12_target]
x12_ho_y <- x12_ho[, x12_target]
x12_removals <- c(x12_target, "target", "id", "ps_ind_14", "ps_car_03_cat") #
x12_train_bal <- x12_train_bal[, setdiff(names(x12_train_bal), x12_removals)]
x12_ho <- x12_ho[, setdiff(names(x12_ho), x12_removals)]
x12_train_bal_dmat <- xgb.DMatrix(as.matrix(x12_train_bal), label=x12_train_bal_y)
x12_ho_dmat <- xgb.DMatrix(as.matrix(x12_ho), label=x12_ho_y)
x12_params <-  list(
    "objective" = "binary:logistic",
    "eval_metric" = "logloss",  
    # "objective" = "multi:softprob",
    # "eval_metric" = "mlogloss",
    # "num_class" = length(unique(x12_train_bal_y)),  # <-- 8 classes including "0"
    "eta" = 0.2,  # 0.4 is way too slow
    "max_depth" = 5,  # experimenting with turning the max depth way down
    "subsample" = 0.8,
    "colsample_bytree" = 0.5,
    "lambda" = 0,
    "alpha" = 1,
    "gamma" = 0,
    "max_delta_step" = 1,
    "scale_pos_weight" = 10,
    "nthread" = 4)
# x12_xgb_cv <- xgb.cv(
#     data= x12_train_bal_dmat,
#     nfold=5,
#     params=x12_params,
#     nrounds=4000,
#     early_stopping_rounds = 50,
#     print_every_n = 1)  
# x12_nrounds <- which.max(x12_xgb_cv$evaluation_log$test_auc_mean)   # 3430 is when it converged, but it might be too good?
# x12_nrounds <- which.min(x12_xgb_cv$evaluation_log$test_logloss_mean)
x12_nrounds <- 25
x12_xgb <- xgboost::xgboost(
    data=x12_train_bal_dmat,
    nrounds=x12_nrounds,
    params = x12_params,
    print_every_n=1,
    save_name = NULL,
    save_period = NULL)
x12_feat_imp <- xgboost::xgb.importance(feature_names = names(x12_train_bal), model=x12_xgb)
xgboost::xgb.plot.importance(x12_feat_imp[1:20,])
x12_preds <- data.frame(actual=x12_ho_y, preds=predict(x12_xgb, x12_ho_dmat))
ggplot(x12_preds, aes(x=preds, fill=as.factor(actual))) +
    geom_density(alpha=0.5, position='identity') +
    ggtitle(x12_nrounds)


saveRDS(x12_xgb, file=paste0("cache/feat_targeting/ft_mods/ft_x12_", x12_target,  "_mod.rds"))
saveRDS(names(x12_train_bal), file=paste0("cache/feat_targeting/ft_feats/ft_x12_", x12_target,  "_feats.rds"))
rm(list=ls()[grepl("^x12_", ls())])




# x13 --------------------------
# target: ps_ind_13
# type: binary class
# removals: target, id, 
set.seed(this_seed)
x13_target <- "ps_ind_13_bin"
x13_samp_size <- 1000
table(train[, x13_target])

x13_train <- data.frame(train)
table(x13_train[, x13_target])
x13_train <- x13_train[x13_train[, x13_target] >= 0, ]

x13_ho_indx <- caret::createDataPartition(y=x13_train[, x13_target], times=1, p=0.25, list=F)
x13_ho <- x13_train[x13_ho_indx, ]
x13_train <- x13_train[-x13_ho_indx, ]

assert_that(all(sapply(x13_ho, function(x) sum(is.na(x))) == 0))

# fix and rebalance -- we can probably use much smaller samples
x13_train_bal <- rbind(
    x13_train[x13_train[, x13_target] == 0, ] %>% sample_n(x13_samp_size*10, replace=nrow(.) < x13_samp_size),
    x13_train[x13_train[, x13_target] == 1, ] %>% sample_n(x13_samp_size, replace=nrow(.) < x13_samp_size))

table(x13_train_bal[, x13_target])
x13_train_bal_y <- x13_train_bal[, x13_target]
x13_ho_y <- x13_ho[, x13_target]
x13_removals <- c(x13_target, "target", "id", "ps_ind_14", "ps_car_03_cat", "ps_ind_16_bin") #
x13_train_bal <- x13_train_bal[, setdiff(names(x13_train_bal), x13_removals)]
x13_ho <- x13_ho[, setdiff(names(x13_ho), x13_removals)]
x13_train_bal_dmat <- xgb.DMatrix(as.matrix(x13_train_bal), label=x13_train_bal_y)
x13_ho_dmat <- xgb.DMatrix(as.matrix(x13_ho), label=x13_ho_y)
x13_params <-  list(
    "objective" = "binary:logistic",
    "eval_metric" = "logloss",  
    # "objective" = "multi:softprob",
    # "eval_metric" = "mlogloss",
    # "num_class" = length(unique(x13_train_bal_y)),  # <-- 8 classes including "0"
    "eta" = 0.2,  # 0.4 is way too slow
    "max_depth" = 5,  # experimenting with turning the max depth way down
    "subsample" = 0.8,
    "colsample_bytree" = 0.5,
    "lambda" = 0,
    "alpha" = 1,
    "gamma" = 0,
    "max_delta_step" = 1,
    "scale_pos_weight" = 10,
    "nthread" = 4)
# x13_xgb_cv <- xgb.cv(
#     data= x13_train_bal_dmat,
#     nfold=5,
#     params=x13_params,
#     nrounds=4000,
#     early_stopping_rounds = 50,
#     print_every_n = 1)
# x13_nrounds <- which.max(x13_xgb_cv$evaluation_log$test_auc_mean)   # 3430 is when it converged, but it might be too good?
# x13_nrounds <- which.min(x13_xgb_cv$evaluation_log$test_logloss_mean)
x13_nrounds <- 15
x13_xgb <- xgboost::xgboost(
    data=x13_train_bal_dmat,
    nrounds=x13_nrounds,
    params = x13_params,
    print_every_n=1,
    save_name = NULL,
    save_period = NULL)
x13_feat_imp <- xgboost::xgb.importance(feature_names = names(x13_train_bal), model=x13_xgb)
xgboost::xgb.plot.importance(x13_feat_imp[1:20,])
x13_preds <- data.frame(actual=x13_ho_y, preds=predict(x13_xgb, x13_ho_dmat))
ggplot(x13_preds, aes(x=preds, fill=as.factor(actual))) +
    geom_density(alpha=0.5, position='identity') +
    ggtitle(x13_nrounds)


saveRDS(x13_xgb, file=paste0("cache/feat_targeting/ft_mods/ft_x13_", x13_target,  "_mod.rds"))
saveRDS(names(x13_train_bal), file=paste0("cache/feat_targeting/ft_feats/ft_x13_", x13_target,  "_feats.rds"))
rm(list=ls()[grepl("^x13_", ls())])






# x14 --------------------------
# target: ps_ind_14
# type: 7 multi class after removal of -1
# removals: target, id, 
set.seed(this_seed)
x14_target <- "ps_ind_14"
x14_samp_size <- 1000
table(train[, x14_target])

x14_train <- data.frame(train)
table(x14_train[, x14_target])

x14_ho_indx <- caret::createDataPartition(y=x14_train[, x14_target], times=1, p=0.25, list=F)
x14_ho <- x14_train[x14_ho_indx, ]
x14_train <- x14_train[-x14_ho_indx, ]

assert_that(all(sapply(x14_ho, function(x) sum(is.na(x))) == 0))

# fix and rebalance -- we can probably use much smaller samples
x14_ho[, x14_target] %>% table()
table(x14_train[, x14_target])
x14_train_bal <- rbind(
    x14_train[x14_train[, x14_target] == 0, ] %>% sample_n(x14_samp_size, replace=nrow(.) < x14_samp_size),
    x14_train[x14_train[, x14_target] == 1, ] %>% sample_n(x14_samp_size, replace=nrow(.) < x14_samp_size),
    x14_train[x14_train[, x14_target] == 2, ] %>% sample_n(x14_samp_size, replace=nrow(.) < x14_samp_size),
    x14_train[x14_train[, x14_target] == 3, ] %>% sample_n(x14_samp_size, replace=nrow(.) < x14_samp_size))
    # x14_train[x14_train[, x14_target] == 4, ] %>% sample_n(x14_samp_size, replace=nrow(.) < x14_samp_size))
        
        #' not enough value "4" categories here for us to use that...

table(x14_train_bal[, x14_target])
x14_train_bal_y <- x14_train_bal[, x14_target]
x14_ho_y <- x14_ho[, x14_target]
x14_removals <- c(x14_target, "target", "id", "ps_ind_10_bin", "ps_ind_12_bin", "ps_ind_13_bin", "ps_ind_11_bin") 
x14_train_bal <- x14_train_bal[, setdiff(names(x14_train_bal), x14_removals)]
x14_ho <- x14_ho[, setdiff(names(x14_ho), x14_removals)]
x14_train_bal_dmat <- xgb.DMatrix(as.matrix(x14_train_bal), label=x14_train_bal_y)
x14_ho_dmat <- xgb.DMatrix(as.matrix(x14_ho), label=x14_ho_y)
x14_params <-  list(
    # "objective" = "binary:logistic",
    # "eval_metric" = "auc",  
    "objective" = "multi:softprob",
    "eval_metric" = "mlogloss",
    "num_class" = length(unique(x14_train_bal_y)),  # <-- 8 classes including "0"
    "eta" = 0.25,
    "max_depth" = 5,
    "subsample" = 0.8,
    "colsample_bytree" = 0.5,
    "lambda" = 0,
    "alpha" = 1,
    "gamma" = 0,
    "max_delta_step" = 1,
    "scale_pos_weight" = 1,
    "nthread" = 4)
# x14_xgb_cv <- xgb.cv(
#     data= x14_train_bal_dmat,
#     nfold=5,
#     params=x14_params,
#     nrounds=4000,
#     early_stopping_rounds = 50,
#     print_every_n = 1)  
# x14_nrounds <- which.min(x14_xgb_cv$evaluation_log$test_mlogloss_mean)
x14_nrounds <- 50
x14_xgb <- xgboost::xgboost(
    data=x14_train_bal_dmat,
    nrounds=x14_nrounds,
    params = x14_params,
    print_every_n=1,
    save_name = NULL,
    save_period = NULL)
x14_feat_imp <- xgboost::xgb.importance(feature_names = names(x14_train_bal), model=x14_xgb)
xgboost::xgb.plot.importance(x14_feat_imp[1:20,])
x14_preds <- as.data.frame(matrix(predict(x14_xgb, x14_ho_dmat), ncol=length(unique(x14_train_bal_y)), byrow=T))
names(x14_preds) <- paste0("feat_tar_x14_", x14_target, "_", sprintf("%02.0f", 0:(length(unique(x14_train_bal_y)) - 1))    )
x14_preds <- cbind(x14_ho_y, x14_preds)
x14_preds_gath <- tidyr::gather(x14_preds, pred_cat, pred_val, -x14_ho_y) %>% 
    group_by(x14_ho_y, pred_cat) %>%
    summarise(mean_pred_val = mean(pred_val, na.rm=T)) %>% ungroup
ggplot(x14_preds_gath, aes(x=x14_ho_y, y=pred_cat, fill=mean_pred_val)) +
    geom_tile(color="White", size=0.1) +
    scale_fill_viridis(name="Mean Prediction") +
    coord_equal() + 
    ggtitle(paste0("Mean Predictions: ", x14_target))

# this model is kind of terrible, but this should be alright
saveRDS(x14_xgb, file=paste0("cache/feat_targeting/ft_mods/ft_x14_", x14_target,  "_mod.rds"))
saveRDS(names(x14_train_bal), file=paste0("cache/feat_targeting/ft_feats/ft_x14_", x14_target,  "_feats.rds"))
rm(list=ls()[grepl("^x14_", ls())])



# x15 --------------------------
# target: ps_ind_15
# type: 7 multi class after removal of -1
# removals: target, id, 
set.seed(this_seed)
x15_target <- "ps_ind_15"
x15_samp_size <- 20000
table(train[, x15_target])

x15_train <- data.frame(train)
table(x15_train[, x15_target])

x15_ho_indx <- caret::createDataPartition(y=x15_train[, x15_target], times=1, p=0.25, list=F)
x15_ho <- x15_train[x15_ho_indx, ]
x15_train <- x15_train[-x15_ho_indx, ]

assert_that(all(sapply(x15_ho, function(x) sum(is.na(x))) == 0))

# fix and rebalance -- we can probably use much smaller samples
x15_ho[, x15_target] %>% table()
table(x15_train[, x15_target])
x15_train_bal <- rbind(
    x15_train[x15_train[, x15_target] == 0, ] %>% sample_n(x15_samp_size, replace=nrow(.) < x15_samp_size),
    x15_train[x15_train[, x15_target] == 1, ] %>% sample_n(x15_samp_size, replace=nrow(.) < x15_samp_size),
    x15_train[x15_train[, x15_target] == 2, ] %>% sample_n(x15_samp_size, replace=nrow(.) < x15_samp_size),
    x15_train[x15_train[, x15_target] == 3, ] %>% sample_n(x15_samp_size, replace=nrow(.) < x15_samp_size),
    x15_train[x15_train[, x15_target] == 4, ] %>% sample_n(x15_samp_size, replace=nrow(.) < x15_samp_size),
    x15_train[x15_train[, x15_target] == 5, ] %>% sample_n(x15_samp_size, replace=nrow(.) < x15_samp_size),
    x15_train[x15_train[, x15_target] == 6, ] %>% sample_n(x15_samp_size, replace=nrow(.) < x15_samp_size),
    x15_train[x15_train[, x15_target] == 7, ] %>% sample_n(x15_samp_size, replace=nrow(.) < x15_samp_size),
    x15_train[x15_train[, x15_target] == 8, ] %>% sample_n(x15_samp_size, replace=nrow(.) < x15_samp_size),
    x15_train[x15_train[, x15_target] == 9, ] %>% sample_n(x15_samp_size, replace=nrow(.) < x15_samp_size),
    x15_train[x15_train[, x15_target] == 10, ] %>% sample_n(x15_samp_size, replace=nrow(.) < x15_samp_size),
    x15_train[x15_train[, x15_target] == 11, ] %>% sample_n(x15_samp_size, replace=nrow(.) < x15_samp_size),
    x15_train[x15_train[, x15_target] == 12, ] %>% sample_n(x15_samp_size, replace=nrow(.) < x15_samp_size),
    x15_train[x15_train[, x15_target] == 13, ] %>% sample_n(x15_samp_size, replace=nrow(.) < x15_samp_size))
    
table(x15_train_bal[, x15_target])
x15_train_bal_y <- x15_train_bal[, x15_target]
x15_ho_y <- x15_ho[, x15_target]
x15_removals <- c(x15_target, "target", "id", "ps_ind_18_bin", "ps_ind_16_bin", "ps_ind_03") 
x15_train_bal <- x15_train_bal[, setdiff(names(x15_train_bal), x15_removals)]
x15_ho <- x15_ho[, setdiff(names(x15_ho), x15_removals)]
x15_train_bal_dmat <- xgb.DMatrix(as.matrix(x15_train_bal), label=x15_train_bal_y)
x15_ho_dmat <- xgb.DMatrix(as.matrix(x15_ho), label=x15_ho_y)
x15_params <-  list(
    # "objective" = "binary:logistic",
    # "eval_metric" = "auc",  
    "objective" = "multi:softprob",
    "eval_metric" = "mlogloss",
    "num_class" = length(unique(x15_train_bal_y)),  # <-- 8 classes including "0"
    "eta" = 0.25,
    "max_depth" = 6,       # might need to boost this up if the predictions suck too badly
    "subsample" = 0.8,
    "colsample_bytree" = 0.5,
    "lambda" = 0,
    "alpha" = 1,
    "gamma" = 0,
    "max_delta_step" = 1,
    "scale_pos_weight" = 1,
    "nthread" = 4)
# x15_xgb_cv <- xgb.cv(
#     data= x15_train_bal_dmat,
#     nfold=5,
#     params=x15_params,
#     nrounds=4000,
#     early_stopping_rounds = 50,
#     print_every_n = 1)
# x15_nrounds <- which.min(x15_xgb_cv$evaluation_log$test_mlogloss_mean)
x15_nrounds <- 200
x15_xgb <- xgboost::xgboost(
    data=x15_train_bal_dmat,
    nrounds=x15_nrounds,
    params = x15_params,
    print_every_n=1,
    save_name = NULL,
    save_period = NULL)
x15_feat_imp <- xgboost::xgb.importance(feature_names = names(x15_train_bal), model=x15_xgb)
xgboost::xgb.plot.importance(x15_feat_imp[1:20,])
x15_preds <- as.data.frame(matrix(predict(x15_xgb, x15_ho_dmat), ncol=length(unique(x15_train_bal_y)), byrow=T))
names(x15_preds) <- paste0("feat_tar_x15_", x15_target, "_", sprintf("%02.0f", 0:(length(unique(x15_train_bal_y)) - 1))    )
x15_preds <- cbind(x15_ho_y, x15_preds)
x15_preds_gath <- tidyr::gather(x15_preds, pred_cat, pred_val, -x15_ho_y) %>% 
    group_by(x15_ho_y, pred_cat) %>%
    summarise(mean_pred_val = mean(pred_val, na.rm=T)) %>% ungroup
ggplot(x15_preds_gath, aes(x=x15_ho_y, y=pred_cat, fill=mean_pred_val)) +
    geom_tile(color="White", size=0.1) +
    scale_fill_viridis(name="Mean Prediction") +
    coord_equal() + 
    ggtitle(paste0("Mean Predictions: ", x15_target, " with ", x15_nrounds, " nrounds"))

# this model is kind of terrible, but this should be alright
saveRDS(x15_xgb, file=paste0("cache/feat_targeting/ft_mods/ft_x15_", x15_target,  "_mod.rds"))
saveRDS(names(x15_train_bal), file=paste0("cache/feat_targeting/ft_feats/ft_x15_", x15_target,  "_feats.rds"))
rm(list=ls()[grepl("^x15_", ls())])



