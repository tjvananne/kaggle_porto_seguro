
#' I need to split this up into trainA and trainB for each feature stack so I can
#' build level 2 models off of these



# Level 1 feature targeting:
source("r_scripts/feat_targeting/feat_targeting_config.R")
train <- readRDS("input/train.rds")

# fp for file path
fp_ft_modsA <- "cache/feat_targeting/ft_modsA/"
fp_ft_modsB <- "cache/feat_targeting/ft_modsB/"
fp_ft_modsAB <- "cache/feat_targeting/ft_modsAB/"
fp_ft_feats <- "cache/feat_targeting/ft_feats/"
fp_ft_remaining_feats <- "cache/feat_targeting/all_remaining_feats.rds"

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

# if these dirs dont exist, then create them
if(!dir.exists(fp_ft_modsA)) {dir.create(fp_ft_modsA)}
if(!dir.exists(fp_ft_modsB)) {dir.create(fp_ft_modsB)}
if(!dir.exists(fp_ft_modsAB)) {dir.create(fp_ft_modsAB)}
if(!dir.exists(fp_ft_feats)) {dir.create(fp_ft_feats)}

# write out feature targeting progress -- update this file after every feature
if(!file.exists(fp_ft_remaining_feats)) {
    all_feats <- names(train)[!grepl("^id$", names(train)) & !grepl("^target$", names(train))]
    saveRDS(all_feats, "cache/feat_targeting/all_remaining_feats.rds")
} 

this_seed <- 1776





# X1 --------------------------
# target: ps_ind_01
# type: 8 class multi
# removals: target, id, 
set.seed(this_seed)
x1_rem_feats <- readRDS(fp_ft_remaining_feats)
x1_target <- "ps_ind_01"
table(train[, x1_target])
x1_samp_size <- 10000


# classes should start at zero
x1_trainA <- data.frame(trainA)
x1_trainB <- data.frame(trainB)

x1_ho_indx <- caret::createDataPartition(y=trainA[, x1_target], times=1, p=0.10, list=F)
x1_ho <- rbind(x1_trainA[x1_ho_indx, ], x1_trainB[x1_ho_indx, ])
x1_trainA <- x1_trainA[-x1_ho_indx, ]
x1_trainB <- x1_trainB[-x1_ho_indx, ]

    # unique target values should be the same across A/B
    assert_that(all(intersect(unique(x1_trainA[, x1_target]), unique(x1_trainB[, x1_target])) %in% x1_trainA[, x1_target]))
    assert_that(all(intersect(unique(x1_trainA[, x1_target]), unique(x1_trainB[, x1_target])) %in% x1_trainB[, x1_target]))

x1_trainA_bal <- take_equal_sample(x1_trainA, x1_samp_size, x1_target, unique(x1_trainA[, x1_target]))
x1_trainB_bal <- take_equal_sample(x1_trainB, x1_samp_size, x1_target, unique(x1_trainB[, x1_target]))
x1_train_bal <- rbind(x1_trainA_bal, x1_trainB_bal)    

x1_trainA_bal_y <- x1_trainA_bal[, x1_target]
x1_trainB_bal_y <- x1_trainB_bal[, x1_target]
x1_train_bal_y <- c(x1_trainA_bal_y, x1_trainB_bal_y)
x1_ho_y <- x1_ho[, x1_target]
x1_removals <- c(x1_target, "target", "id", "ps_car_05_cat")

x1_trainA_bal <- x1_trainA_bal[, setdiff(names(x1_trainA_bal), x1_removals)]
x1_trainB_bal <- x1_trainB_bal[, setdiff(names(x1_trainB_bal), x1_removals)]
x1_train_bal <- x1_train_bal[, setdiff(names(x1_train_bal), x1_removals)]
x1_ho <- x1_ho[, setdiff(names(x1_ho), x1_removals)]

x1_trainA_bal_dmat <- xgb.DMatrix(as.matrix(x1_trainA_bal), label=x1_trainA_bal_y)
x1_trainB_bal_dmat <- xgb.DMatrix(as.matrix(x1_trainB_bal), label=x1_trainB_bal_y)
x1_train_bal_dmat <- xgb.DMatrix(as.matrix(x1_train_bal), label=x1_train_bal_y)
x1_ho_dmat <- xgb.DMatrix(as.matrix(x1_ho), label=x1_ho_y)

x1_params <-  list(
    # "objective" = "binary:logistic",
    # "eval_metric" = "auc",  
    "objective" = "multi:softprob",
    "eval_metric" = "mlogloss",
    "num_class" = length(unique(x1_trainA_bal_y)),  # <-- 8 classes including "0"
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


# first train the combined model
x1_nrounds <- 40
x1_xgb <- xgboost::xgboost(
    nrounds=x1_nrounds,
    # nrounds=30,        # for additional training rounds on already trained model
    # xgb_model=x1_xgb,  # for additional training rounds on already trained model
    
    data=x1_train_bal_dmat,
    params=x1_params,
    print_every_n=1, save_period = NULL, save_name = NULL)


# feature importance for cheater removal
x1_feat_imp <- xgboost::xgb.importance(feature_names = names(x1_train_bal), model=x1_xgb)
xgboost::xgb.plot.importance(x1_feat_imp[1:20,])


# control flow for which plotting route to take based on the type of model 
if(x1_params$objective == "multi:softprob") {
    # multi classification
    x1_preds <- as.data.frame(matrix(predict(x1_xgb, x1_ho_dmat), ncol=length(unique(x1_train_bal_y)), byrow=T))
    names(x1_preds) <- paste0("feat_tar_x1_", x1_target, "_", 0:(length(unique(x2_trainA_bal_y)) - 1))
    x1_preds <- cbind(x1_ho_y, x1_preds)
    x1_preds_gath <- tidyr::gather(x1_preds, pred_cat, pred_val, -x1_ho_y) %>% 
        group_by(x1_ho_y, pred_cat) %>%
        summarise(mean_pred_val = mean(pred_val, na.rm=T)) %>% ungroup
    ggplot(x1_preds_gath, aes(x=x1_ho_y, y=pred_cat, fill=mean_pred_val)) +
        geom_tile(color="White", size=0.1) +
        scale_fill_viridis(name="Mean Prediction") +
        coord_equal() + 
        ggtitle(paste0("Mean Predictions: ", x1_target, " after ", x1_xgb$niter, " nrounds"))
} else if(x1_params$objective == "binary:logistic") {
    # binary classification -- not tested yet
    x1_preds <- data.frame(actual=x1_ho_y, preds=predict(x1_xgb, x1_ho_dmat))
    ggplot(x1_preds, aes(x=preds, fill=as.factor(actual))) +
        geom_histogram(alpha=0.5, position='identity') +
        ggtitle(paste0("Prediction density: ", x1_target, " after ", x1_xgb$niter, " nrounds"))
} else if(x1_params$objective == "reg:linear") {
    # linear regression -- not tested yet
    x1_preds <- data.frame(actual=x1_ho_y, preds=predict(x1_xgb, x1_ho_dmat))
    x1_preds$residual <- x1_preds$actual - x1_preds$preds
    ggplot(x1_preds, x=actual, y=residual) +
        geom_point(alpha=0.4) + 
        ggtitle(paste0("Prediction residuals: ", x1_target, " after ", x1_xgb$niter, " nrounds"))
}
    

# if results above look good, train model B and the combined AB model
x1_xgbB <- xgboost::xgboost(
    data=x1_trainB_bal_dmat,
    nrounds=x1_xgb$niter,    # <-- safer in case we add more rounds manually
    params = x1_params,
    print_every_n=1, save_period = NULL, save_name = NULL)


# train model A first
x1_xgbA <- xgboost::xgboost(
    data=x1_trainA_bal_dmat,
    nrounds=x1_xgb$niter,
    params=x1_params,
    print_every_n=1, save_period = NULL, save_name = NULL)

# write it all out    
saveRDS(x1_xgbA, file=paste0(fp_ft_modsA, "ft_x1_", x1_target, "_mod.rds"))
saveRDS(x1_xgbB, file=paste0(fp_ft_modsB, "ft_x1_", x1_target, "_mod.rds"))
saveRDS(x1_xgb, file=paste0(fp_ft_modsAB, "ft_x1_", x1_target, "_mod.rds"))
saveRDS(names(x1_train_bal), file=paste0(fp_ft_feats, "ft_x1_", x1_target, "_mod.rds"))

# update remaining features
x1_rem_feats <- readRDS(fp_ft_remaining_feats)
x1_rem_feats <- setdiff(x1_rem_feats, x1_target)
saveRDS(x1_rem_feats, fp_ft_remaining_feats)

rm(list=ls()[grepl("^x1_", ls())])
gc()




# x2 --------------------------
# target: ps_ind_02_cat
# type: 8 class multi
# removals: target, id, 
set.seed(this_seed)
x2_rem_feats <- readRDS(fp_ft_remaining_feats)
x2_target <- "ps_ind_02_cat"
table(train[, x2_target])
x2_samp_size <- 20000


# classes should start at zero
x2_trainA <- data.frame(trainA)
x2_trainB <- data.frame(trainB)
    

# do this before fixing target variable
x2_ho_indx <- caret::createDataPartition(y=trainA[, x2_target], times=1, p=0.10, list=F)
x2_ho <- rbind(x2_trainA[x2_ho_indx, ], x2_trainB[x2_ho_indx, ])
x2_trainA <- x2_trainA[-x2_ho_indx, ]
x2_trainB <- x2_trainB[-x2_ho_indx, ]

    # X-two specific, fixing the target variable to look how we want it to
    x2_trainA[, x2_target] <- x2_trainA[, x2_target] - 1
    x2_trainB[, x2_target] <- x2_trainB[, x2_target] - 1
    x2_ho[, x2_target] <- x2_ho[, x2_target] - 1
    x2_trainA <- x2_trainA[x2_trainA[, x2_target] >= 0, ]
    x2_trainB <- x2_trainB[x2_trainB[, x2_target] >= 0, ]
    table(x2_trainA[, x2_target])
    table(x2_trainB[, x2_target])


# unique target values should be the same across A/B
assert_that(all(intersect(unique(x2_trainA[, x2_target]), unique(x2_trainB[, x2_target])) %in% x2_trainA[, x2_target]))
assert_that(all(intersect(unique(x2_trainA[, x2_target]), unique(x2_trainB[, x2_target])) %in% x2_trainB[, x2_target]))

    
x2_trainA_bal <- take_equal_sample(x2_trainA, x2_samp_size, x2_target, unique(x2_trainA[, x2_target]))
x2_trainB_bal <- take_equal_sample(x2_trainB, x2_samp_size, x2_target, unique(x2_trainB[, x2_target]))
x2_train_bal <- rbind(x2_trainA_bal, x2_trainB_bal)    
table(x2_trainA_bal[, x2_target])


x2_trainA_bal_y <- x2_trainA_bal[, x2_target]
x2_trainB_bal_y <- x2_trainB_bal[, x2_target]
x2_train_bal_y <- c(x2_trainA_bal_y, x2_trainB_bal_y)
x2_ho_y <- x2_ho[, x2_target]
x2_removals <- c(x2_target, "target", "id", "ps_ind_03", "ps_ind_15", "ps_ind_04_cat")

x2_trainA_bal <- x2_trainA_bal[, setdiff(names(x2_trainA_bal), x2_removals)]
x2_trainB_bal <- x2_trainB_bal[, setdiff(names(x2_trainB_bal), x2_removals)]
x2_train_bal <- x2_train_bal[, setdiff(names(x2_train_bal), x2_removals)]
x2_ho <- x2_ho[, setdiff(names(x2_ho), x2_removals)]

x2_trainA_bal_dmat <- xgb.DMatrix(as.matrix(x2_trainA_bal), label=x2_trainA_bal_y)
x2_trainB_bal_dmat <- xgb.DMatrix(as.matrix(x2_trainB_bal), label=x2_trainB_bal_y)
x2_train_bal_dmat <- xgb.DMatrix(as.matrix(x2_train_bal), label=x2_train_bal_y)
x2_ho_dmat <- xgb.DMatrix(as.matrix(x2_ho), label=x2_ho_y)

x2_params <-  list(
    # "objective" = "binary:logistic",
    # "eval_metric" = "auc",  
    "objective" = "multi:softprob",
    "eval_metric" = "mlogloss",
    "num_class" = length(unique(x2_trainA_bal_y)),  # <-- 8 classes including "0"
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


# first train the combined model
x2_nrounds <- 10
x2_xgb <- xgboost::xgboost(
    nrounds=x2_nrounds,
    # nrounds=90,        # for additional training rounds on already trained model
    # xgb_model=x2_xgb,  # for additional training rounds on already trained model
    
    data=x2_train_bal_dmat,
    params=x2_params,
    print_every_n=1, save_period = NULL, save_name = NULL)


# feature importance for cheater removal
x2_feat_imp <- xgboost::xgb.importance(feature_names = names(x2_train_bal), model=x2_xgb)
xgboost::xgb.plot.importance(x2_feat_imp[1:20,])


# control flow for which plotting route to take based on the type of model 
if(x2_params$objective == "multi:softprob") {
    # multi classification
    x2_preds <- as.data.frame(matrix(predict(x2_xgb, x2_ho_dmat), ncol=length(unique(x2_train_bal_y)), byrow=T))
    names(x2_preds) <- paste0("feat_tar_x2_", x2_target, "_", 0:(length(unique(x2_trainA_bal_y)) - 1))
    x2_preds <- cbind(x2_ho_y, x2_preds)
    x2_preds_gath <- tidyr::gather(x2_preds, pred_cat, pred_val, -x2_ho_y) %>% 
        group_by(x2_ho_y, pred_cat) %>%
        summarise(mean_pred_val = mean(pred_val, na.rm=T)) %>% ungroup
    ggplot(x2_preds_gath, aes(x=x2_ho_y, y=pred_cat, fill=mean_pred_val)) +
        geom_tile(color="White", size=0.1) +
        scale_fill_viridis(name="Mean Prediction") +
        coord_equal() + 
        ggtitle(paste0("Mean Predictions: ", x2_target, " after ", x2_xgb$niter, " nrounds"))
} else if(x2_params$objective == "binary:logistic") {
    # binary classification -- not tested yet
    x2_preds <- data.frame(actual=x2_ho_y, preds=predict(x2_xgb, x2_ho_dmat))
    ggplot(x2_preds, aes(x=preds, fill=as.factor(actual))) +
        geom_histogram(alpha=0.5, position='identity') +
        ggtitle(paste0("Prediction density: ", x2_target, " after ", x2_xgb$niter, " nrounds"))
} else if(x2_params$objective == "reg:linear") {
    # linear regression -- not tested yet
    x2_preds <- data.frame(actual=x2_ho_y, preds=predict(x2_xgb, x2_ho_dmat))
    x2_preds$residual <- x2_preds$actual - x2_preds$preds
    ggplot(x2_preds, x=actual, y=residual) +
        geom_point(alpha=0.4) + 
        ggtitle(paste0("Prediction residuals: ", x2_target, " after ", x2_xgb$niter, " nrounds"))
}

# in depth analysis of multi classi
x2_preds_manual_gath <- tidyr::gather(x2_preds, pred_cat, pred_val, -x2_ho_y)
x2_ho_y_val <- 3
ggplot(data=x2_preds_manual_gath %>% filter(x2_ho_y == x2_ho_y_val), aes(x=pred_val, fill=as.factor(pred_cat))) +
    geom_histogram(alpha=0.3, position='identity') +
    ggtitle(paste0("Investigating where ho_y = ", x2_ho_y_val, " and nrounds = ", x2_xgb$niter))




# if results above look good, train model B and the combined AB model
x2_xgbB <- xgboost::xgboost(
    data=x2_trainB_bal_dmat,
    nrounds=x2_xgb$niter,    # <-- safer in case we add more rounds manually
    params = x2_params,
    print_every_n=1, save_period = NULL, save_name = NULL)


# train model A first
x2_xgbA <- xgboost::xgboost(
    data=x2_trainA_bal_dmat,
    nrounds=x2_xgb$niter,
    params=x2_params,
    print_every_n=1, save_period = NULL, save_name = NULL)

# write it all out    
saveRDS(x2_xgbA, file=paste0(fp_ft_modsA, "ft_x2_", x2_target, "_mod.rds"))
saveRDS(x2_xgbB, file=paste0(fp_ft_modsB, "ft_x2_", x2_target, "_mod.rds"))
saveRDS(x2_xgb, file=paste0(fp_ft_modsAB, "ft_x2_", x2_target, "_mod.rds"))
saveRDS(names(x2_train_bal), file=paste0(fp_ft_feats, "ft_x2_", x2_target, "_mod.rds"))

# update remaining features
x2_rem_feats <- readRDS(fp_ft_remaining_feats)
x2_rem_feats <- setdiff(x2_rem_feats, x2_target)
saveRDS(x2_rem_feats, fp_ft_remaining_feats)

rm(list=ls()[grepl("^x2_", ls())])
gc()


# x3 --------------------------
# target: ps_ind_03
# type: 8 class multi
# removals: target, id, 
set.seed(this_seed)
x3_rem_feats <- readRDS(fp_ft_remaining_feats)
x3_target <- "ps_ind_03"
table(train[, x3_target])
x3_samp_size <- 8000


# classes should start at zero
x3_trainA <- data.frame(trainA)
x3_trainB <- data.frame(trainB)


# do this before fixing target variable
x3_ho_indx <- caret::createDataPartition(y=trainA[, x3_target], times=1, p=0.10, list=F)
x3_ho <- rbind(x3_trainA[x3_ho_indx, ], x3_trainB[x3_ho_indx, ])
x3_trainA <- x3_trainA[-x3_ho_indx, ]
x3_trainB <- x3_trainB[-x3_ho_indx, ]

# fix target value here if necessary

# unique target values should be the same across A/B
assert_that(all(intersect(unique(x3_trainA[, x3_target]), unique(x3_trainB[, x3_target])) %in% x3_trainA[, x3_target]))
assert_that(all(intersect(unique(x3_trainA[, x3_target]), unique(x3_trainB[, x3_target])) %in% x3_trainB[, x3_target]))

# take equalized samples of each of the target variable
x3_trainA_bal <- take_equal_sample(x3_trainA, x3_samp_size, x3_target, unique(x3_trainA[, x3_target]))
x3_trainB_bal <- take_equal_sample(x3_trainB, x3_samp_size, x3_target, unique(x3_trainB[, x3_target]))
x3_train_bal <- rbind(x3_trainA_bal, x3_trainB_bal)    
table(x3_trainA_bal[, x3_target])


x3_trainA_bal_y <- x3_trainA_bal[, x3_target]
x3_trainB_bal_y <- x3_trainB_bal[, x3_target]
x3_train_bal_y <- c(x3_trainA_bal_y, x3_trainB_bal_y)
x3_ho_y <- x3_ho[, x3_target]
x3_removals <- c(x3_target, "target", "id", "ps_ind_02_cat", "ps_ind_05_cat", "ps_ind_15", "ps_car_07_cat",
                 "ps_ind_01", "ps_ind_18_bin", "ps_ind_16_bin")   # huge gains after removals up to this point!
                 # "ps_car_01_cat", "ps_ind_04_cat", "ps_ind_09_bin", "ps_car_05_cat", "ps_car_13", "ps_car_14",
                 # "ps_car_12", "ps_car_11_cat", "ps_car_11", "ps_reg_03", "ps_ind_07_bin", "ps_car_15", "ps_reg_02", "ps_car_03_cat", # top right drop off
                 # "ps_car_04_cat", "ps_car_06_cat", "ps_car_09_cat", "ps_reg_01", "ps_car_02_cat")

x3_trainA_bal <- x3_trainA_bal[, setdiff(names(x3_trainA_bal), x3_removals)]
x3_trainB_bal <- x3_trainB_bal[, setdiff(names(x3_trainB_bal), x3_removals)]
x3_train_bal <- x3_train_bal[, setdiff(names(x3_train_bal), x3_removals)]
x3_ho <- x3_ho[, setdiff(names(x3_ho), x3_removals)]

x3_trainA_bal_dmat <- xgb.DMatrix(as.matrix(x3_trainA_bal), label=x3_trainA_bal_y)
x3_trainB_bal_dmat <- xgb.DMatrix(as.matrix(x3_trainB_bal), label=x3_trainB_bal_y)
x3_train_bal_dmat <- xgb.DMatrix(as.matrix(x3_train_bal), label=x3_train_bal_y)
x3_ho_dmat <- xgb.DMatrix(as.matrix(x3_ho), label=x3_ho_y)

x3_params <-  list(
    # "objective" = "binary:logistic",
    # "eval_metric" = "auc",  
    "objective" = "multi:softprob",
    "eval_metric" = "mlogloss",
    "num_class" = length(unique(x3_trainA_bal_y)),  # <-- 8 classes including "0"
    "eta" = 0.4,
    "max_depth" = 5,
    "subsample" = 0.8,
    "colsample_bytree" = 0.5,
    "lambda" = 0,
    "alpha" = 1,
    "gamma" = 0,
    "max_delta_step" = 1,
    "scale_pos_weight" = 1,
    "nthread" = 4) # 131 best nrounds


# first train the combined model
x3_nrounds <- 90
x3_xgb <- xgboost::xgboost(
    nrounds=x3_nrounds,
    # nrounds=30, xgb_model=x3_xgb,  # for additional training rounds on already trained model
    data=x3_train_bal_dmat,
    params=x3_params,
    print_every_n=1, save_period = NULL, save_name = NULL)


{
# feature importance for cheater removal
x3_feat_imp <- xgboost::xgb.importance(feature_names = names(x3_train_bal), model=x3_xgb)
xgboost::xgb.plot.importance(x3_feat_imp[1:20,])


# control flow for which plotting route to take based on the type of model 
if(x3_params$objective == "multi:softprob") {
    # multi classification
    x3_preds <- as.data.frame(matrix(predict(x3_xgb, x3_ho_dmat), ncol=length(unique(x3_train_bal_y)), byrow=T))
    names(x3_preds) <- paste0("feat_tar_x3_", x3_target, "_", sprintf("%02.0f", 0:(length(unique(x3_trainA_bal_y)) - 1)))
    x3_preds <- cbind(x3_ho_y, x3_preds)
    x3_preds_gath <- tidyr::gather(x3_preds, pred_cat, pred_val, -x3_ho_y) %>% 
        group_by(x3_ho_y, pred_cat) %>%
        summarise(mean_pred_val = mean(pred_val, na.rm=T)) %>% ungroup
    ggplot(x3_preds_gath, aes(x=x3_ho_y, y=pred_cat, fill=mean_pred_val)) +
        geom_tile(color="White", size=0.1) +
        # scale_fill_viridis(name="Mean Prediction") +
        scale_fill_viridis(name="Mean Prediction", limits=c(0.001, 0.16)) +
        coord_equal() +  
        ggtitle(paste0("Mean Predictions: ", x3_target, " after ", x3_xgb$niter, " nrounds"))
} else if(x3_params$objective == "binary:logistic") {
    # binary classification -- not tested yet
    x3_preds <- data.frame(actual=x3_ho_y, preds=predict(x3_xgb, x3_ho_dmat))
    ggplot(x3_preds, aes(x=preds, fill=as.factor(actual))) +
        geom_histogram(alpha=0.5, position='identity') +
        ggtitle(paste0("Prediction density: ", x3_target, " after ", x3_xgb$niter, " nrounds"))
} else if(x3_params$objective == "reg:linear") {
    # linear regression -- not tested yet
    x3_preds <- data.frame(actual=x3_ho_y, preds=predict(x3_xgb, x3_ho_dmat))
    x3_preds$residual <- x3_preds$actual - x3_preds$preds
    ggplot(x3_preds, x=actual, y=residual) +
        geom_point(alpha=0.4) + 
        ggtitle(paste0("Prediction residuals: ", x3_target, " after ", x3_xgb$niter, " nrounds"))
}
}  # feature importance and model plot


# in depth analysis of multi classi
x3_preds_manual_gath <- tidyr::gather(x3_preds, pred_cat, pred_val, -x3_ho_y)
x3_ho_y_val <- 6
ggplot(data=x3_preds_manual_gath %>% filter(x3_ho_y == x3_ho_y_val), aes(x=pred_val, fill=as.factor(pred_cat))) +
    geom_histogram(alpha=0.3, position='identity') +
    ggtitle(paste0("Investigating where ho_y = ", x3_ho_y_val, " and nrounds = ", x3_xgb$niter))




# if results above look good, train model B and the combined AB model
x3_xgbB <- xgboost::xgboost(
    data=x3_trainB_bal_dmat,
    nrounds=x3_xgb$niter,    # <-- safer in case we add more rounds manually
    params = x3_params,
    print_every_n=1, save_period = NULL, save_name = NULL)


# train model A first
x3_xgbA <- xgboost::xgboost(
    data=x3_trainA_bal_dmat,
    nrounds=x3_xgb$niter,
    params=x3_params,
    print_every_n=1, save_period = NULL, save_name = NULL)

# write it all out    
saveRDS(x3_xgbA, file=paste0(fp_ft_modsA, "ft_x3_", x3_target, "_mod.rds"))
saveRDS(x3_xgbB, file=paste0(fp_ft_modsB, "ft_x3_", x3_target, "_mod.rds"))
saveRDS(x3_xgb, file=paste0(fp_ft_modsAB, "ft_x3_", x3_target, "_mod.rds"))
saveRDS(names(x3_train_bal), file=paste0(fp_ft_feats, "ft_x3_", x3_target, "_mod.rds"))

# update remaining features
x3_rem_feats <- readRDS(fp_ft_remaining_feats)
x3_rem_feats <- setdiff(x3_rem_feats, x3_target)
saveRDS(x3_rem_feats, fp_ft_remaining_feats)

rm(list=ls()[grepl("^x3_", ls())])
gc()






# x4 --------------------------
# target: ps_ind_04_cat
# type: 8 class multi
# removals: target, id, 
set.seed(this_seed)
x4_rem_feats <- readRDS(fp_ft_remaining_feats)
x4_target <- "ps_ind_04_cat"
table(train[, x4_target])
x4_samp_size <- 8000


# classes should start at zero
x4_trainA <- data.frame(trainA)
x4_trainB <- data.frame(trainB)


# do this before fixing target variable
x4_ho_indx <- caret::createDataPartition(y=trainA[, x4_target], times=1, p=0.10, list=F)
x4_ho <- rbind(x4_trainA[x4_ho_indx, ], x4_trainB[x4_ho_indx, ])
x4_trainA <- x4_trainA[-x4_ho_indx, ]
x4_trainB <- x4_trainB[-x4_ho_indx, ]

# fix target value here if necessary
x4_trainA <- x4_trainA[x4_trainA[, x4_target] >= 0,]
x4_trainB <- x4_trainB[x4_trainB[, x4_target] >= 0, ]
table(x4_trainA[, x4_target])
table(x4_trainB[, x4_target])


# unique target values should be the same across A/B
assert_that(all(intersect(unique(x4_trainA[, x4_target]), unique(x4_trainB[, x4_target])) %in% x4_trainA[, x4_target]))
assert_that(all(intersect(unique(x4_trainA[, x4_target]), unique(x4_trainB[, x4_target])) %in% x4_trainB[, x4_target]))

# take equalized samples of each of the target variable
x4_trainA_bal <- take_equal_sample(x4_trainA, x4_samp_size, x4_target, unique(x4_trainA[, x4_target]))
x4_trainB_bal <- take_equal_sample(x4_trainB, x4_samp_size, x4_target, unique(x4_trainB[, x4_target]))
x4_train_bal <- rbind(x4_trainA_bal, x4_trainB_bal)    
table(x4_trainA_bal[, x4_target])


x4_trainA_bal_y <- x4_trainA_bal[, x4_target]
x4_trainB_bal_y <- x4_trainB_bal[, x4_target]
x4_train_bal_y <- c(x4_trainA_bal_y, x4_trainB_bal_y)
x4_ho_y <- x4_ho[, x4_target]
x4_removals <- c(x4_target, "target", "id", "ps_ind_07_bin")


x4_trainA_bal <- x4_trainA_bal[, setdiff(names(x4_trainA_bal), x4_removals)]
x4_trainB_bal <- x4_trainB_bal[, setdiff(names(x4_trainB_bal), x4_removals)]
x4_train_bal <- x4_train_bal[, setdiff(names(x4_train_bal), x4_removals)]
x4_ho <- x4_ho[, setdiff(names(x4_ho), x4_removals)]

x4_trainA_bal_dmat <- xgb.DMatrix(as.matrix(x4_trainA_bal), label=x4_trainA_bal_y)
x4_trainB_bal_dmat <- xgb.DMatrix(as.matrix(x4_trainB_bal), label=x4_trainB_bal_y)
x4_train_bal_dmat <- xgb.DMatrix(as.matrix(x4_train_bal), label=x4_train_bal_y)
x4_ho_dmat <- xgb.DMatrix(as.matrix(x4_ho), label=x4_ho_y)

x4_params <-  list(
    "objective" = "binary:logistic",
    "eval_metric" = "auc",  
    # "objective" = "multi:softprob",
    # "eval_metric" = "mlogloss",
    # "num_class" = length(unique(x4_trainA_bal_y)),  # <-- 8 classes including "0"
    "eta" = 0.4,
    "max_depth" = 5,
    "subsample" = 0.8,
    "colsample_bytree" = 0.5,
    "lambda" = 0,
    "alpha" = 1,
    "gamma" = 0,
    "max_delta_step" = 1,
    "scale_pos_weight" = 1,
    "nthread" = 4) # 131 best nrounds


# first train the combined model
x4_nrounds <- 100
x4_xgb <- xgboost::xgboost(
    nrounds=x4_nrounds,
    # nrounds=15, xgb_model=x4_xgb,  # for additional training rounds on already trained model
    data=x4_train_bal_dmat,
    params=x4_params,
    print_every_n=1, save_period = NULL, save_name = NULL)


{
    # feature importance for cheater removal
    x4_feat_imp <- xgboost::xgb.importance(feature_names = names(x4_train_bal), model=x4_xgb)
    xgboost::xgb.plot.importance(x4_feat_imp[1:20,])
    
    
    # control flow for which plotting route to take based on the type of model 
    if(x4_params$objective == "multi:softprob") {
        # multi classification
        x4_preds <- as.data.frame(matrix(predict(x4_xgb, x4_ho_dmat), ncol=length(unique(x4_train_bal_y)), byrow=T))
        names(x4_preds) <- paste0("feat_tar_x4_", x4_target, "_", sprintf("%02.0f", 0:(length(unique(x4_trainA_bal_y)) - 1)))
        x4_preds <- cbind(x4_ho_y, x4_preds)
        x4_preds_gath <- tidyr::gather(x4_preds, pred_cat, pred_val, -x4_ho_y) %>% 
            group_by(x4_ho_y, pred_cat) %>%
            summarise(mean_pred_val = mean(pred_val, na.rm=T)) %>% ungroup
        ggplot(x4_preds_gath, aes(x=x4_ho_y, y=pred_cat, fill=mean_pred_val)) +
            geom_tile(color="White", size=0.1) +
            # scale_fill_viridis(name="Mean Prediction") +
            scale_fill_viridis(name="Mean Prediction", limits=c(0.001, 0.16)) +
            coord_equal() +  
            ggtitle(paste0("Mean Predictions: ", x4_target, " after ", x4_xgb$niter, " nrounds"))
    } else if(x4_params$objective == "binary:logistic") {
        # binary classification -- not tested yet
        x4_preds <- data.frame(actual=x4_ho_y, preds=predict(x4_xgb, x4_ho_dmat))
        ggplot(x4_preds, aes(x=preds, fill=as.factor(actual))) +
            geom_histogram(alpha=0.5, position='identity') +
            ggtitle(paste0("Prediction density: ", x4_target, " after ", x4_xgb$niter, " nrounds"))
    } else if(x4_params$objective == "reg:linear") {
        # linear regression -- not tested yet
        x4_preds <- data.frame(actual=x4_ho_y, preds=predict(x4_xgb, x4_ho_dmat))
        x4_preds$residual <- x4_preds$actual - x4_preds$preds
        ggplot(x4_preds, x=actual, y=residual) +
            geom_point(alpha=0.4) + 
            ggtitle(paste0("Prediction residuals: ", x4_target, " after ", x4_xgb$niter, " nrounds"))
    }
}  # feature importance and model plot


# in depth analysis of multi classi
x4_preds_manual_gath <- tidyr::gather(x4_preds, pred_cat, pred_val, -x4_ho_y)
x4_ho_y_val <- 6
ggplot(data=x4_preds_manual_gath %>% filter(x4_ho_y == x4_ho_y_val), aes(x=pred_val, fill=as.factor(pred_cat))) +
    geom_histogram(alpha=0.3, position='identity') +
    ggtitle(paste0("Investigating where ho_y = ", x4_ho_y_val, " and nrounds = ", x4_xgb$niter))




# if results above look good, train model B and the combined AB model
x4_xgbB <- xgboost::xgboost(
    data=x4_trainB_bal_dmat,
    nrounds=x4_xgb$niter,    # <-- safer in case we add more rounds manually
    params = x4_params,
    print_every_n=1, save_period = NULL, save_name = NULL)


# train model A first
x4_xgbA <- xgboost::xgboost(
    data=x4_trainA_bal_dmat,
    nrounds=x4_xgb$niter,
    params=x4_params,
    print_every_n=1, save_period = NULL, save_name = NULL)

# write it all out    
saveRDS(x4_xgbA, file=paste0(fp_ft_modsA, "ft_x4_", x4_target, "_mod.rds"))
saveRDS(x4_xgbB, file=paste0(fp_ft_modsB, "ft_x4_", x4_target, "_mod.rds"))
saveRDS(x4_xgb, file=paste0(fp_ft_modsAB, "ft_x4_", x4_target, "_mod.rds"))
saveRDS(names(x4_train_bal), file=paste0(fp_ft_feats, "ft_x4_", x4_target, "_mod.rds"))

# update remaining features
x4_rem_feats <- readRDS(fp_ft_remaining_feats)
x4_rem_feats <- setdiff(x4_rem_feats, x4_target)
saveRDS(x4_rem_feats, fp_ft_remaining_feats)

rm(list=ls()[grepl("^x4_", ls())])
gc()



# x5 --------------------------
# target: ps_ind_04_cat
# type: 8 class multi
# removals: target, id, 
set.seed(this_seed)
x5_rem_feats <- readRDS(fp_ft_remaining_feats)
x5_target <- "ps_ind_05_cat"
table(train[, x5_target])
x5_samp_size <- 8000


# classes should start at zero
x5_trainA <- data.frame(trainA)
x5_trainB <- data.frame(trainB)


# do this before fixing target variable
x5_ho_indx <- caret::createDataPartition(y=trainA[, x5_target], times=1, p=0.10, list=F)
x5_ho <- rbind(x5_trainA[x5_ho_indx, ], x5_trainB[x5_ho_indx, ])
x5_trainA <- x5_trainA[-x5_ho_indx, ]
x5_trainB <- x5_trainB[-x5_ho_indx, ]

# fix target value here if necessary
x5_trainA <- x5_trainA[x5_trainA[, x5_target] >= 0,]
x5_trainB <- x5_trainB[x5_trainB[, x5_target] >= 0, ]
table(x5_trainA[, x5_target])
table(x5_trainB[, x5_target])


# unique target values should be the same across A/B
assert_that(all(intersect(unique(x5_trainA[, x5_target]), unique(x5_trainB[, x5_target])) %in% x5_trainA[, x5_target]))
assert_that(all(intersect(unique(x5_trainA[, x5_target]), unique(x5_trainB[, x5_target])) %in% x5_trainB[, x5_target]))

# take equalized samples of each of the target variable
x5_trainA_bal <- take_equal_sample(x5_trainA, x5_samp_size, x5_target, unique(x5_trainA[, x5_target]))
x5_trainB_bal <- take_equal_sample(x5_trainB, x5_samp_size, x5_target, unique(x5_trainB[, x5_target]))
x5_train_bal <- rbind(x5_trainA_bal, x5_trainB_bal)    
table(x5_trainA_bal[, x5_target])


x5_trainA_bal_y <- x5_trainA_bal[, x5_target]
x5_trainB_bal_y <- x5_trainB_bal[, x5_target]
x5_train_bal_y <- c(x5_trainA_bal_y, x5_trainB_bal_y)
x5_ho_y <- x5_ho[, x5_target]
x5_removals <- c(x5_target, "target", "id", "ps_ind_03")


x5_trainA_bal <- x5_trainA_bal[, setdiff(names(x5_trainA_bal), x5_removals)]
x5_trainB_bal <- x5_trainB_bal[, setdiff(names(x5_trainB_bal), x5_removals)]
x5_train_bal <- x5_train_bal[, setdiff(names(x5_train_bal), x5_removals)]
x5_ho <- x5_ho[, setdiff(names(x5_ho), x5_removals)]

x5_trainA_bal_dmat <- xgb.DMatrix(as.matrix(x5_trainA_bal), label=x5_trainA_bal_y)
x5_trainB_bal_dmat <- xgb.DMatrix(as.matrix(x5_trainB_bal), label=x5_trainB_bal_y)
x5_train_bal_dmat <- xgb.DMatrix(as.matrix(x5_train_bal), label=x5_train_bal_y)
x5_ho_dmat <- xgb.DMatrix(as.matrix(x5_ho), label=x5_ho_y)

x5_params <-  list(
    # "objective" = "binary:logistic",
    # "eval_metric" = "auc",  
    "objective" = "multi:softprob",
    "eval_metric" = "mlogloss",
    "num_class" = length(unique(x5_trainA_bal_y)),  # <-- 8 classes including "0"
    "eta" = 0.4,
    "max_depth" = 5,
    "subsample" = 0.8,
    "colsample_bytree" = 0.5,
    "lambda" = 0,
    "alpha" = 1,
    "gamma" = 0,
    "max_delta_step" = 1,
    "scale_pos_weight" = 1,
    "nthread" = 4) # 131 best nrounds


# first train the combined model
x5_nrounds <- 70
x5_xgb <- xgboost::xgboost(
    nrounds=x5_nrounds,
    # nrounds=10, xgb_model=x5_xgb,  # for additional training rounds on already trained model
    data=x5_train_bal_dmat,
    params=x5_params,
    print_every_n=1, save_period = NULL, save_name = NULL)


{
    # feature importance for cheater removal
    x5_feat_imp <- xgboost::xgb.importance(feature_names = names(x5_train_bal), model=x5_xgb)
    xgboost::xgb.plot.importance(x5_feat_imp[1:20,])
    
    
    # control flow for which plotting route to take based on the type of model 
    if(x5_params$objective == "multi:softprob") {
        # multi classification
        x5_preds <- as.data.frame(matrix(predict(x5_xgb, x5_ho_dmat), ncol=length(unique(x5_train_bal_y)), byrow=T))
        names(x5_preds) <- paste0("feat_tar_x5_", x5_target, "_", sprintf("%02.0f", 0:(length(unique(x5_trainA_bal_y)) - 1)))
        x5_preds <- cbind(x5_ho_y, x5_preds)
        x5_preds_gath <- tidyr::gather(x5_preds, pred_cat, pred_val, -x5_ho_y) %>% 
            group_by(x5_ho_y, pred_cat) %>%
            summarise(mean_pred_val = mean(pred_val, na.rm=T)) %>% ungroup
        ggplot(x5_preds_gath, aes(x=x5_ho_y, y=pred_cat, fill=mean_pred_val)) +
            geom_tile(color="White", size=0.1) +
            # scale_fill_viridis(name="Mean Prediction") +
            scale_fill_viridis(name="Mean Prediction", limits=c(0.001, (2* (1/length(unique(x5_train_bal_y)))))) +
            coord_equal() +  
            ggtitle(paste0("Mean Predictions: ", x5_target, " after ", x5_xgb$niter, " nrounds"))
    } else if(x5_params$objective == "binary:logistic") {
        # binary classification -- not tested yet
        x5_preds <- data.frame(actual=x5_ho_y, preds=predict(x5_xgb, x5_ho_dmat))
        ggplot(x5_preds, aes(x=preds, fill=as.factor(actual))) +
            geom_histogram(alpha=0.5, position='identity') +
            ggtitle(paste0("Prediction density: ", x5_target, " after ", x5_xgb$niter, " nrounds"))
    } else if(x5_params$objective == "reg:linear") {
        # linear regression -- not tested yet
        x5_preds <- data.frame(actual=x5_ho_y, preds=predict(x5_xgb, x5_ho_dmat))
        x5_preds$residual <- x5_preds$actual - x5_preds$preds
        ggplot(x5_preds, x=actual, y=residual) +
            geom_point(alpha=0.4) + 
            ggtitle(paste0("Prediction residuals: ", x5_target, " after ", x5_xgb$niter, " nrounds"))
    }
}  # feature importance and model plot


# in depth analysis of multi classi
x5_preds_manual_gath <- tidyr::gather(x5_preds, pred_cat, pred_val, -x5_ho_y)
x5_ho_y_val <- 3
ggplot(data=x5_preds_manual_gath %>% filter(x5_ho_y == x5_ho_y_val), aes(x=pred_val, fill=as.factor(pred_cat))) +
    geom_histogram(alpha=0.3, position='identity') +
    ggtitle(paste0("Investigating where ho_y = ", x5_ho_y_val, " and nrounds = ", x5_xgb$niter))




# if results above look good, train model B and the combined AB model
x5_xgbB <- xgboost::xgboost(
    data=x5_trainB_bal_dmat,
    nrounds=x5_xgb$niter,    # <-- safer in case we add more rounds manually
    params = x5_params,
    print_every_n=1, save_period = NULL, save_name = NULL)


# train model A first
x5_xgbA <- xgboost::xgboost(
    data=x5_trainA_bal_dmat,
    nrounds=x5_xgb$niter,
    params=x5_params,
    print_every_n=1, save_period = NULL, save_name = NULL)

# write it all out    
saveRDS(x5_xgbA, file=paste0(fp_ft_modsA, "ft_x5_", x5_target, "_mod.rds"))
saveRDS(x5_xgbB, file=paste0(fp_ft_modsB, "ft_x5_", x5_target, "_mod.rds"))
saveRDS(x5_xgb, file=paste0(fp_ft_modsAB, "ft_x5_", x5_target, "_mod.rds"))
saveRDS(names(x5_train_bal), file=paste0(fp_ft_feats, "ft_x5_", x5_target, "_mod.rds"))

# update remaining features
x5_rem_feats <- readRDS(fp_ft_remaining_feats)
x5_rem_feats <- setdiff(x5_rem_feats, x5_target)
saveRDS(x5_rem_feats, fp_ft_remaining_feats)

rm(list=ls()[grepl("^x5_", ls())])
gc()


# x6 --------------------------
# target: ps_ind_04_cat
# type: 8 class multi
# removals: target, id, 
set.seed(this_seed)
x6_rem_feats <- readRDS(fp_ft_remaining_feats)
x6_target <- "ps_ind_06_bin"
table(train[, x6_target])
x6_samp_size <- 50000


# classes should start at zero
x6_trainA <- data.frame(trainA)
x6_trainB <- data.frame(trainB)


# do this before fixing target variable
x6_ho_indx <- caret::createDataPartition(y=trainA[, x6_target], times=1, p=0.10, list=F)
x6_ho <- rbind(x6_trainA[x6_ho_indx, ], x6_trainB[x6_ho_indx, ])
x6_trainA <- x6_trainA[-x6_ho_indx, ]
x6_trainB <- x6_trainB[-x6_ho_indx, ]

# fix target value here if necessary
x6_trainA <- x6_trainA[x6_trainA[, x6_target] >= 0,]
x6_trainB <- x6_trainB[x6_trainB[, x6_target] >= 0, ]
table(x6_trainA[, x6_target])
table(x6_trainB[, x6_target])


# unique target values should be the same across A/B
assert_that(all(intersect(unique(x6_trainA[, x6_target]), unique(x6_trainB[, x6_target])) %in% x6_trainA[, x6_target]))
assert_that(all(intersect(unique(x6_trainA[, x6_target]), unique(x6_trainB[, x6_target])) %in% x6_trainB[, x6_target]))

# take equalized samples of each of the target variable
x6_trainA_bal <- take_equal_sample(x6_trainA, x6_samp_size, x6_target, unique(x6_trainA[, x6_target]))
x6_trainB_bal <- take_equal_sample(x6_trainB, x6_samp_size, x6_target, unique(x6_trainB[, x6_target]))
x6_train_bal <- rbind(x6_trainA_bal, x6_trainB_bal)    
table(x6_trainA_bal[, x6_target])


x6_trainA_bal_y <- x6_trainA_bal[, x6_target]
x6_trainB_bal_y <- x6_trainB_bal[, x6_target]
x6_train_bal_y <- c(x6_trainA_bal_y, x6_trainB_bal_y)
x6_ho_y <- x6_ho[, x6_target]
x6_removals <- c(x6_target, "target", "id", "ps_ind_07_bin", "ps_ind_08_bin", "ps_ind_09_bin")


x6_trainA_bal <- x6_trainA_bal[, setdiff(names(x6_trainA_bal), x6_removals)]
x6_trainB_bal <- x6_trainB_bal[, setdiff(names(x6_trainB_bal), x6_removals)]
x6_train_bal <- x6_train_bal[, setdiff(names(x6_train_bal), x6_removals)]
x6_ho <- x6_ho[, setdiff(names(x6_ho), x6_removals)]

x6_trainA_bal_dmat <- xgb.DMatrix(as.matrix(x6_trainA_bal), label=x6_trainA_bal_y)
x6_trainB_bal_dmat <- xgb.DMatrix(as.matrix(x6_trainB_bal), label=x6_trainB_bal_y)
x6_train_bal_dmat <- xgb.DMatrix(as.matrix(x6_train_bal), label=x6_train_bal_y)
x6_ho_dmat <- xgb.DMatrix(as.matrix(x6_ho), label=x6_ho_y)

x6_params <-  list(
    "objective" = "binary:logistic",
    "eval_metric" = "auc",  
    # "objective" = "multi:softprob",
    # "eval_metric" = "mlogloss",
    # "num_class" = length(unique(x6_trainA_bal_y)),  # <-- 8 classes including "0"
    "eta" = 0.4,
    "max_depth" = 5,
    "subsample" = 0.8,
    "colsample_bytree" = 0.5,
    "lambda" = 0,
    "alpha" = 1,
    "gamma" = 0,
    "max_delta_step" = 1,
    "scale_pos_weight" = 1,
    "nthread" = 4) # 131 best nrounds


# first train the combined model
x6_nrounds <- 45
x6_xgb <- xgboost::xgboost(
    nrounds=x6_nrounds,
    # nrounds=10, xgb_model=x6_xgb,  # for additional training rounds on already trained model
    data=x6_train_bal_dmat,
    params=x6_params,
    print_every_n=1, save_period = NULL, save_name = NULL)


{
    # feature importance for cheater removal
    x6_feat_imp <- xgboost::xgb.importance(feature_names = names(x6_train_bal), model=x6_xgb)
    xgboost::xgb.plot.importance(x6_feat_imp[1:20,])
    
    
    # control flow for which plotting route to take based on the type of model 
    if(x6_params$objective == "multi:softprob") {
        # multi classification
        x6_preds <- as.data.frame(matrix(predict(x6_xgb, x6_ho_dmat), ncol=length(unique(x6_train_bal_y)), byrow=T))
        names(x6_preds) <- paste0("feat_tar_x6_", x6_target, "_", sprintf("%02.0f", 0:(length(unique(x6_trainA_bal_y)) - 1)))
        x6_preds <- cbind(x6_ho_y, x6_preds)
        x6_preds_gath <- tidyr::gather(x6_preds, pred_cat, pred_val, -x6_ho_y) %>% 
            group_by(x6_ho_y, pred_cat) %>%
            summarise(mean_pred_val = mean(pred_val, na.rm=T)) %>% ungroup
        ggplot(x6_preds_gath, aes(x=x6_ho_y, y=pred_cat, fill=mean_pred_val)) +
            geom_tile(color="White", size=0.1) +
            # scale_fill_viridis(name="Mean Prediction") +
            scale_fill_viridis(name="Mean Prediction", limits=c(0.001, (2* (1/length(unique(x6_train_bal_y)))))) +
            coord_equal() +  
            ggtitle(paste0("Mean Predictions: ", x6_target, " after ", x6_xgb$niter, " nrounds"))
    } else if(x6_params$objective == "binary:logistic") {
        # binary classification -- not tested yet
        x6_preds <- data.frame(actual=x6_ho_y, preds=predict(x6_xgb, x6_ho_dmat))
        ggplot(x6_preds, aes(x=preds, fill=as.factor(actual))) +
            geom_histogram(alpha=0.5, position='identity') +
            ggtitle(paste0("Prediction density: ", x6_target, " after ", x6_xgb$niter, " nrounds"))
    } else if(x6_params$objective == "reg:linear") {
        # linear regression -- not tested yet
        x6_preds <- data.frame(actual=x6_ho_y, preds=predict(x6_xgb, x6_ho_dmat))
        x6_preds$residual <- x6_preds$actual - x6_preds$preds
        ggplot(x6_preds, x=actual, y=residual) +
            geom_point(alpha=0.4) + 
            ggtitle(paste0("Prediction residuals: ", x6_target, " after ", x6_xgb$niter, " nrounds"))
    }
}  # feature importance and model plot


# in depth analysis of multi classi
x6_preds_manual_gath <- tidyr::gather(x6_preds, pred_cat, pred_val, -x6_ho_y)
x6_ho_y_val <- 3
ggplot(data=x6_preds_manual_gath %>% filter(x6_ho_y == x6_ho_y_val), aes(x=pred_val, fill=as.factor(pred_cat))) +
    geom_histogram(alpha=0.3, position='identity') +
    ggtitle(paste0("Investigating where ho_y = ", x6_ho_y_val, " and nrounds = ", x6_xgb$niter))




# if results above look good, train model B and the combined AB model
x6_xgbB <- xgboost::xgboost(
    data=x6_trainB_bal_dmat,
    nrounds=x6_xgb$niter,    # <-- safer in case we add more rounds manually
    params = x6_params,
    print_every_n=1, save_period = NULL, save_name = NULL)


# train model A first
x6_xgbA <- xgboost::xgboost(
    data=x6_trainA_bal_dmat,
    nrounds=x6_xgb$niter,
    params=x6_params,
    print_every_n=1, save_period = NULL, save_name = NULL)

# write it all out    
saveRDS(x6_xgbA, file=paste0(fp_ft_modsA, "ft_x6_", x6_target, "_mod.rds"))
saveRDS(x6_xgbB, file=paste0(fp_ft_modsB, "ft_x6_", x6_target, "_mod.rds"))
saveRDS(x6_xgb, file=paste0(fp_ft_modsAB, "ft_x6_", x6_target, "_mod.rds"))
saveRDS(names(x6_train_bal), file=paste0(fp_ft_feats, "ft_x6_", x6_target, "_mod.rds"))

# update remaining features
x6_rem_feats <- readRDS(fp_ft_remaining_feats)
x6_rem_feats <- setdiff(x6_rem_feats, x6_target)
saveRDS(x6_rem_feats, fp_ft_remaining_feats)

rm(list=ls()[grepl("^x6_", ls())])
gc()



# x7 --------------------------
# target: ps_ind_04_cat
# type: 8 class multi
# removals: target, id, 
set.seed(this_seed)
x7_rem_feats <- readRDS(fp_ft_remaining_feats)
x7_target <- "ps_ind_07_bin"
table(train[, x7_target])
x7_samp_size <- 40000


# classes should start at zero
x7_trainA <- data.frame(trainA)
x7_trainB <- data.frame(trainB)


# do this before fixing target variable
x7_ho_indx <- caret::createDataPartition(y=trainA[, x7_target], times=1, p=0.10, list=F)
x7_ho <- rbind(x7_trainA[x7_ho_indx, ], x7_trainB[x7_ho_indx, ])
x7_trainA <- x7_trainA[-x7_ho_indx, ]
x7_trainB <- x7_trainB[-x7_ho_indx, ]

# fix target value here if necessary
x7_trainA <- x7_trainA[x7_trainA[, x7_target] >= 0,]
x7_trainB <- x7_trainB[x7_trainB[, x7_target] >= 0, ]
table(x7_trainA[, x7_target])
table(x7_trainB[, x7_target])


# unique target values should be the same across A/B
assert_that(all(intersect(unique(x7_trainA[, x7_target]), unique(x7_trainB[, x7_target])) %in% x7_trainA[, x7_target]))
assert_that(all(intersect(unique(x7_trainA[, x7_target]), unique(x7_trainB[, x7_target])) %in% x7_trainB[, x7_target]))

# take equalized samples of each of the target variable
x7_trainA_bal <- take_equal_sample(x7_trainA, x7_samp_size, x7_target, unique(x7_trainA[, x7_target]))
x7_trainB_bal <- take_equal_sample(x7_trainB, x7_samp_size, x7_target, unique(x7_trainB[, x7_target]))
x7_train_bal <- rbind(x7_trainA_bal, x7_trainB_bal)    
table(x7_trainA_bal[, x7_target])


x7_trainA_bal_y <- x7_trainA_bal[, x7_target]
x7_trainB_bal_y <- x7_trainB_bal[, x7_target]
x7_train_bal_y <- c(x7_trainA_bal_y, x7_trainB_bal_y)
x7_ho_y <- x7_ho[, x7_target]
x7_removals <- c(x7_target, "target", "id", "ps_ind_08_bin", "ps_ind_09_bin", "ps_ind_06_bin")


x7_trainA_bal <- x7_trainA_bal[, setdiff(names(x7_trainA_bal), x7_removals)]
x7_trainB_bal <- x7_trainB_bal[, setdiff(names(x7_trainB_bal), x7_removals)]
x7_train_bal <- x7_train_bal[, setdiff(names(x7_train_bal), x7_removals)]
x7_ho <- x7_ho[, setdiff(names(x7_ho), x7_removals)]

x7_trainA_bal_dmat <- xgb.DMatrix(as.matrix(x7_trainA_bal), label=x7_trainA_bal_y)
x7_trainB_bal_dmat <- xgb.DMatrix(as.matrix(x7_trainB_bal), label=x7_trainB_bal_y)
x7_train_bal_dmat <- xgb.DMatrix(as.matrix(x7_train_bal), label=x7_train_bal_y)
x7_ho_dmat <- xgb.DMatrix(as.matrix(x7_ho), label=x7_ho_y)

x7_params <-  list(
    "objective" = "binary:logistic",
    "eval_metric" = "auc",  
    # "objective" = "multi:softprob",
    # "eval_metric" = "mlogloss",
    # "num_class" = length(unique(x7_trainA_bal_y)),  # <-- 8 classes including "0"
    "eta" = 0.4,
    "max_depth" = 5,
    "subsample" = 0.8,
    "colsample_bytree" = 0.5,
    "lambda" = 0,
    "alpha" = 1,
    "gamma" = 0,
    "max_delta_step" = 1,
    "scale_pos_weight" = 1,
    "nthread" = 4) # 131 best nrounds


# first train the combined model
x7_nrounds <- 40
x7_xgb <- xgboost::xgboost(
    nrounds=x7_nrounds,
    # nrounds=10, xgb_model=x7_xgb,  # for additional training rounds on already trained model
    data=x7_train_bal_dmat,
    params=x7_params,
    print_every_n=1, save_period = NULL, save_name = NULL)


{
    # feature importance for cheater removal
    x7_feat_imp <- xgboost::xgb.importance(feature_names = names(x7_train_bal), model=x7_xgb)
    xgboost::xgb.plot.importance(x7_feat_imp[1:20,])
    
    
    # control flow for which plotting route to take based on the type of model 
    if(x7_params$objective == "multi:softprob") {
        # multi classification
        x7_preds <- as.data.frame(matrix(predict(x7_xgb, x7_ho_dmat), ncol=length(unique(x7_train_bal_y)), byrow=T))
        names(x7_preds) <- paste0("feat_tar_x7_", x7_target, "_", sprintf("%02.0f", 0:(length(unique(x7_trainA_bal_y)) - 1)))
        x7_preds <- cbind(x7_ho_y, x7_preds)
        x7_preds_gath <- tidyr::gather(x7_preds, pred_cat, pred_val, -x7_ho_y) %>% 
            group_by(x7_ho_y, pred_cat) %>%
            summarise(mean_pred_val = mean(pred_val, na.rm=T)) %>% ungroup
        ggplot(x7_preds_gath, aes(x=x7_ho_y, y=pred_cat, fill=mean_pred_val)) +
            geom_tile(color="White", size=0.1) +
            # scale_fill_viridis(name="Mean Prediction") +
            scale_fill_viridis(name="Mean Prediction", limits=c(0.001, (2* (1/length(unique(x7_train_bal_y)))))) +
            coord_equal() +  
            ggtitle(paste0("Mean Predictions: ", x7_target, " after ", x7_xgb$niter, " nrounds"))
    } else if(x7_params$objective == "binary:logistic") {
        # binary classification -- not tested yet
        x7_preds <- data.frame(actual=x7_ho_y, preds=predict(x7_xgb, x7_ho_dmat))
        ggplot(x7_preds, aes(x=preds, fill=as.factor(actual))) +
            geom_histogram(alpha=0.5, position='identity') +
            ggtitle(paste0("Prediction density: ", x7_target, " after ", x7_xgb$niter, " nrounds"))
    } else if(x7_params$objective == "reg:linear") {
        # linear regression -- not tested yet
        x7_preds <- data.frame(actual=x7_ho_y, preds=predict(x7_xgb, x7_ho_dmat))
        x7_preds$residual <- x7_preds$actual - x7_preds$preds
        ggplot(x7_preds, x=actual, y=residual) +
            geom_point(alpha=0.4) + 
            ggtitle(paste0("Prediction residuals: ", x7_target, " after ", x7_xgb$niter, " nrounds"))
    }
}  # feature importance and model plot


# in depth analysis of multi classi
x7_preds_manual_gath <- tidyr::gather(x7_preds, pred_cat, pred_val, -x7_ho_y)
x7_ho_y_val <- 3
ggplot(data=x7_preds_manual_gath %>% filter(x7_ho_y == x7_ho_y_val), aes(x=pred_val, fill=as.factor(pred_cat))) +
    geom_histogram(alpha=0.3, position='identity') +
    ggtitle(paste0("Investigating where ho_y = ", x7_ho_y_val, " and nrounds = ", x7_xgb$niter))


# if results above look good, train model B and the combined AB model
x7_xgbB <- xgboost::xgboost(
    data=x7_trainB_bal_dmat,
    nrounds=x7_xgb$niter,    # <-- safer in case we add more rounds manually
    params = x7_params,
    print_every_n=1, save_period = NULL, save_name = NULL)


# train model A first
x7_xgbA <- xgboost::xgboost(
    data=x7_trainA_bal_dmat,
    nrounds=x7_xgb$niter,
    params=x7_params,
    print_every_n=1, save_period = NULL, save_name = NULL)

# write it all out    
saveRDS(x7_xgbA, file=paste0(fp_ft_modsA, "ft_x7_", x7_target, "_mod.rds"))
saveRDS(x7_xgbB, file=paste0(fp_ft_modsB, "ft_x7_", x7_target, "_mod.rds"))
saveRDS(x7_xgb, file=paste0(fp_ft_modsAB, "ft_x7_", x7_target, "_mod.rds"))
saveRDS(names(x7_train_bal), file=paste0(fp_ft_feats, "ft_x7_", x7_target, "_mod.rds"))

# update remaining features
x7_rem_feats <- readRDS(fp_ft_remaining_feats)
x7_rem_feats <- setdiff(x7_rem_feats, x7_target)
saveRDS(x7_rem_feats, fp_ft_remaining_feats)

rm(list=ls()[grepl("^x7_", ls())])
gc()



# x8 --------------------------
# target: ps_ind_04_cat
# type: 8 class multi
# removals: target, id, 
set.seed(this_seed)
x8_rem_feats <- readRDS(fp_ft_remaining_feats)
x8_target <- "ps_ind_08_bin"
table(train[, x8_target])
x8_samp_size <- 40000


# classes should start at zero
x8_trainA <- data.frame(trainA)
x8_trainB <- data.frame(trainB)


# do this before fixing target variable
x8_ho_indx <- caret::createDataPartition(y=trainA[, x8_target], times=1, p=0.10, list=F)
x8_ho <- rbind(x8_trainA[x8_ho_indx, ], x8_trainB[x8_ho_indx, ])
x8_trainA <- x8_trainA[-x8_ho_indx, ]
x8_trainB <- x8_trainB[-x8_ho_indx, ]

# fix target value here if necessary
x8_trainA <- x8_trainA[x8_trainA[, x8_target] >= 0,]
x8_trainB <- x8_trainB[x8_trainB[, x8_target] >= 0, ]
table(x8_trainA[, x8_target])
table(x8_trainB[, x8_target])


# unique target values should be the same across A/B
assert_that(all(intersect(unique(x8_trainA[, x8_target]), unique(x8_trainB[, x8_target])) %in% x8_trainA[, x8_target]))
assert_that(all(intersect(unique(x8_trainA[, x8_target]), unique(x8_trainB[, x8_target])) %in% x8_trainB[, x8_target]))

# take equalized samples of each of the target variable
x8_trainA_bal <- take_equal_sample(x8_trainA, x8_samp_size, x8_target, unique(x8_trainA[, x8_target]))
x8_trainB_bal <- take_equal_sample(x8_trainB, x8_samp_size, x8_target, unique(x8_trainB[, x8_target]))
x8_train_bal <- rbind(x8_trainA_bal, x8_trainB_bal)    
table(x8_trainA_bal[, x8_target])


x8_trainA_bal_y <- x8_trainA_bal[, x8_target]
x8_trainB_bal_y <- x8_trainB_bal[, x8_target]
x8_train_bal_y <- c(x8_trainA_bal_y, x8_trainB_bal_y)
x8_ho_y <- x8_ho[, x8_target]
x8_removals <- c(x8_target, "target", "id", "ps_ind_07_bin", "ps_ind_09_bin", "ps_ind_06_bin")


x8_trainA_bal <- x8_trainA_bal[, setdiff(names(x8_trainA_bal), x8_removals)]
x8_trainB_bal <- x8_trainB_bal[, setdiff(names(x8_trainB_bal), x8_removals)]
x8_train_bal <- x8_train_bal[, setdiff(names(x8_train_bal), x8_removals)]
x8_ho <- x8_ho[, setdiff(names(x8_ho), x8_removals)]

x8_trainA_bal_dmat <- xgb.DMatrix(as.matrix(x8_trainA_bal), label=x8_trainA_bal_y)
x8_trainB_bal_dmat <- xgb.DMatrix(as.matrix(x8_trainB_bal), label=x8_trainB_bal_y)
x8_train_bal_dmat <- xgb.DMatrix(as.matrix(x8_train_bal), label=x8_train_bal_y)
x8_ho_dmat <- xgb.DMatrix(as.matrix(x8_ho), label=x8_ho_y)

x8_params <-  list(
    "objective" = "binary:logistic",
    "eval_metric" = "auc",  
    # "objective" = "multi:softprob",
    # "eval_metric" = "mlogloss",
    # "num_class" = length(unique(x8_trainA_bal_y)),  # <-- 8 classes including "0"
    "eta" = 0.4,
    "max_depth" = 5,
    "subsample" = 0.8,
    "colsample_bytree" = 0.5,
    "lambda" = 0,
    "alpha" = 1,
    "gamma" = 0,
    "max_delta_step" = 1,
    "scale_pos_weight" = 1,
    "nthread" = 4) # 131 best nrounds


# first train the combined model
x8_nrounds <- 75
x8_xgb <- xgboost::xgboost(
    nrounds=x8_nrounds,
    # nrounds=10, xgb_model=x8_xgb,  # for additional training rounds on already trained model
    data=x8_train_bal_dmat,
    params=x8_params,
    print_every_n=1, save_period = NULL, save_name = NULL)


{
    # feature importance for cheater removal
    x8_feat_imp <- xgboost::xgb.importance(feature_names = names(x8_train_bal), model=x8_xgb)
    xgboost::xgb.plot.importance(x8_feat_imp[1:20,])
    
    
    # control flow for which plotting route to take based on the type of model 
    if(x8_params$objective == "multi:softprob") {
        # multi classification
        x8_preds <- as.data.frame(matrix(predict(x8_xgb, x8_ho_dmat), ncol=length(unique(x8_train_bal_y)), byrow=T))
        names(x8_preds) <- paste0("feat_tar_x8_", x8_target, "_", sprintf("%02.0f", 0:(length(unique(x8_trainA_bal_y)) - 1)))
        x8_preds <- cbind(x8_ho_y, x8_preds)
        x8_preds_gath <- tidyr::gather(x8_preds, pred_cat, pred_val, -x8_ho_y) %>% 
            group_by(x8_ho_y, pred_cat) %>%
            summarise(mean_pred_val = mean(pred_val, na.rm=T)) %>% ungroup
        ggplot(x8_preds_gath, aes(x=x8_ho_y, y=pred_cat, fill=mean_pred_val)) +
            geom_tile(color="White", size=0.1) +
            # scale_fill_viridis(name="Mean Prediction") +
            scale_fill_viridis(name="Mean Prediction", limits=c(0.001, (2* (1/length(unique(x8_train_bal_y)))))) +
            coord_equal() +  
            ggtitle(paste0("Mean Predictions: ", x8_target, " after ", x8_xgb$niter, " nrounds"))
    } else if(x8_params$objective == "binary:logistic") {
        # binary classification -- not tested yet
        x8_preds <- data.frame(actual=x8_ho_y, preds=predict(x8_xgb, x8_ho_dmat))
        ggplot(x8_preds, aes(x=preds, fill=as.factor(actual))) +
            geom_histogram(alpha=0.5, position='identity') +
            ggtitle(paste0("Prediction density: ", x8_target, " after ", x8_xgb$niter, " nrounds"))
    } else if(x8_params$objective == "reg:linear") {
        # linear regression -- not tested yet
        x8_preds <- data.frame(actual=x8_ho_y, preds=predict(x8_xgb, x8_ho_dmat))
        x8_preds$residual <- x8_preds$actual - x8_preds$preds
        ggplot(x8_preds, x=actual, y=residual) +
            geom_point(alpha=0.4) + 
            ggtitle(paste0("Prediction residuals: ", x8_target, " after ", x8_xgb$niter, " nrounds"))
    }
}  # feature importance and model plot


# in depth analysis of multi classi
x8_preds_manual_gath <- tidyr::gather(x8_preds, pred_cat, pred_val, -x8_ho_y)
x8_ho_y_val <- 3
ggplot(data=x8_preds_manual_gath %>% filter(x8_ho_y == x8_ho_y_val), aes(x=pred_val, fill=as.factor(pred_cat))) +
    geom_histogram(alpha=0.3, position='identity') +
    ggtitle(paste0("Investigating where ho_y = ", x8_ho_y_val, " and nrounds = ", x8_xgb$niter))


# if results above look good, train model B and the combined AB model
x8_xgbB <- xgboost::xgboost(
    data=x8_trainB_bal_dmat,
    nrounds=x8_xgb$niter,    # <-- safer in case we add more rounds manually
    params = x8_params,
    print_every_n=1, save_period = NULL, save_name = NULL)


# train model A first
x8_xgbA <- xgboost::xgboost(
    data=x8_trainA_bal_dmat,
    nrounds=x8_xgb$niter,
    params=x8_params,
    print_every_n=1, save_period = NULL, save_name = NULL)

# write it all out    
saveRDS(x8_xgbA, file=paste0(fp_ft_modsA, "ft_x8_", x8_target, "_mod.rds"))
saveRDS(x8_xgbB, file=paste0(fp_ft_modsB, "ft_x8_", x8_target, "_mod.rds"))
saveRDS(x8_xgb, file=paste0(fp_ft_modsAB, "ft_x8_", x8_target, "_mod.rds"))
saveRDS(names(x8_train_bal), file=paste0(fp_ft_feats, "ft_x8_", x8_target, "_mod.rds"))

# update remaining features
x8_rem_feats <- readRDS(fp_ft_remaining_feats)
x8_rem_feats <- setdiff(x8_rem_feats, x8_target)
saveRDS(x8_rem_feats, fp_ft_remaining_feats)

rm(list=ls()[grepl("^x8_", ls())])
gc()


# x9 --------------------------
# target: ps_ind_04_cat
# type: 8 class multi
# removals: target, id, 
set.seed(this_seed)
x9_rem_feats <- readRDS(fp_ft_remaining_feats)
x9_target <- "ps_ind_09_bin"
table(train[, x9_target])
x9_samp_size <- 50000


# classes should start at zero
x9_trainA <- data.frame(trainA)
x9_trainB <- data.frame(trainB)


# do this before fixing target variable
x9_ho_indx <- caret::createDataPartition(y=trainA[, x9_target], times=1, p=0.10, list=F)
x9_ho <- rbind(x9_trainA[x9_ho_indx, ], x9_trainB[x9_ho_indx, ])
x9_trainA <- x9_trainA[-x9_ho_indx, ]
x9_trainB <- x9_trainB[-x9_ho_indx, ]

# fix target value here if necessary
x9_trainA <- x9_trainA[x9_trainA[, x9_target] >= 0,]
x9_trainB <- x9_trainB[x9_trainB[, x9_target] >= 0, ]
table(x9_trainA[, x9_target])
table(x9_trainB[, x9_target])


# unique target values should be the same across A/B
assert_that(all(intersect(unique(x9_trainA[, x9_target]), unique(x9_trainB[, x9_target])) %in% x9_trainA[, x9_target]))
assert_that(all(intersect(unique(x9_trainA[, x9_target]), unique(x9_trainB[, x9_target])) %in% x9_trainB[, x9_target]))

# take equalized samples of each of the target variable
x9_trainA_bal <- take_equal_sample(x9_trainA, x9_samp_size, x9_target, unique(x9_trainA[, x9_target]))
x9_trainB_bal <- take_equal_sample(x9_trainB, x9_samp_size, x9_target, unique(x9_trainB[, x9_target]))
x9_train_bal <- rbind(x9_trainA_bal, x9_trainB_bal)    
table(x9_trainA_bal[, x9_target])


x9_trainA_bal_y <- x9_trainA_bal[, x9_target]
x9_trainB_bal_y <- x9_trainB_bal[, x9_target]
x9_train_bal_y <- c(x9_trainA_bal_y, x9_trainB_bal_y)
x9_ho_y <- x9_ho[, x9_target]
x9_removals <- c(x9_target, "target", "id", "ps_ind_07_bin", "ps_ind_08_bin", "ps_ind_06_bin")


x9_trainA_bal <- x9_trainA_bal[, setdiff(names(x9_trainA_bal), x9_removals)]
x9_trainB_bal <- x9_trainB_bal[, setdiff(names(x9_trainB_bal), x9_removals)]
x9_train_bal <- x9_train_bal[, setdiff(names(x9_train_bal), x9_removals)]
x9_ho <- x9_ho[, setdiff(names(x9_ho), x9_removals)]

x9_trainA_bal_dmat <- xgb.DMatrix(as.matrix(x9_trainA_bal), label=x9_trainA_bal_y)
x9_trainB_bal_dmat <- xgb.DMatrix(as.matrix(x9_trainB_bal), label=x9_trainB_bal_y)
x9_train_bal_dmat <- xgb.DMatrix(as.matrix(x9_train_bal), label=x9_train_bal_y)
x9_ho_dmat <- xgb.DMatrix(as.matrix(x9_ho), label=x9_ho_y)

x9_params <-  list(
    "objective" = "binary:logistic",
    "eval_metric" = "auc",  
    # "objective" = "multi:softprob",
    # "eval_metric" = "mlogloss",
    # "num_class" = length(unique(x9_trainA_bal_y)),  # <-- 8 classes including "0"
    "eta" = 0.4,
    "max_depth" = 5,
    "subsample" = 0.8,
    "colsample_bytree" = 0.5,
    "lambda" = 0,
    "alpha" = 1,
    "gamma" = 0,
    "max_delta_step" = 1,
    "scale_pos_weight" = 1,
    "nthread" = 4) # 131 best nrounds


# first train the combined model
x9_nrounds <- 50
x9_xgb <- xgboost::xgboost(
    nrounds=x9_nrounds,
    # nrounds=10, xgb_model=x9_xgb,  # for additional training rounds on already trained model
    data=x9_train_bal_dmat,
    params=x9_params,
    print_every_n=1, save_period = NULL, save_name = NULL)


{
    # feature importance for cheater removal
    x9_feat_imp <- xgboost::xgb.importance(feature_names = names(x9_train_bal), model=x9_xgb)
    xgboost::xgb.plot.importance(x9_feat_imp[1:20,])
    
    
    # control flow for which plotting route to take based on the type of model 
    if(x9_params$objective == "multi:softprob") {
        # multi classification
        x9_preds <- as.data.frame(matrix(predict(x9_xgb, x9_ho_dmat), ncol=length(unique(x9_train_bal_y)), byrow=T))
        names(x9_preds) <- paste0("feat_tar_x9_", x9_target, "_", sprintf("%02.0f", 0:(length(unique(x9_trainA_bal_y)) - 1)))
        x9_preds <- cbind(x9_ho_y, x9_preds)
        x9_preds_gath <- tidyr::gather(x9_preds, pred_cat, pred_val, -x9_ho_y) %>% 
            group_by(x9_ho_y, pred_cat) %>%
            summarise(mean_pred_val = mean(pred_val, na.rm=T)) %>% ungroup
        ggplot(x9_preds_gath, aes(x=x9_ho_y, y=pred_cat, fill=mean_pred_val)) +
            geom_tile(color="White", size=0.1) +
            # scale_fill_viridis(name="Mean Prediction") +
            scale_fill_viridis(name="Mean Prediction", limits=c(0.001, (2* (1/length(unique(x9_train_bal_y)))))) +
            coord_equal() +  
            ggtitle(paste0("Mean Predictions: ", x9_target, " after ", x9_xgb$niter, " nrounds"))
    } else if(x9_params$objective == "binary:logistic") {
        # binary classification -- not tested yet
        x9_preds <- data.frame(actual=x9_ho_y, preds=predict(x9_xgb, x9_ho_dmat))
        ggplot(x9_preds, aes(x=preds, fill=as.factor(actual))) +
            geom_histogram(alpha=0.5, position='identity') +
            ggtitle(paste0("Prediction density: ", x9_target, " after ", x9_xgb$niter, " nrounds"))
    } else if(x9_params$objective == "reg:linear") {
        # linear regression -- not tested yet
        x9_preds <- data.frame(actual=x9_ho_y, preds=predict(x9_xgb, x9_ho_dmat))
        x9_preds$residual <- x9_preds$actual - x9_preds$preds
        ggplot(x9_preds, x=actual, y=residual) +
            geom_point(alpha=0.4) + 
            ggtitle(paste0("Prediction residuals: ", x9_target, " after ", x9_xgb$niter, " nrounds"))
    }
}  # feature importance and model plot


# in depth analysis of multi classi
x9_preds_manual_gath <- tidyr::gather(x9_preds, pred_cat, pred_val, -x9_ho_y)
x9_ho_y_val <- 3
ggplot(data=x9_preds_manual_gath %>% filter(x9_ho_y == x9_ho_y_val), aes(x=pred_val, fill=as.factor(pred_cat))) +
    geom_histogram(alpha=0.3, position='identity') +
    ggtitle(paste0("Investigating where ho_y = ", x9_ho_y_val, " and nrounds = ", x9_xgb$niter))


# if results above look good, train model B and the combined AB model
x9_xgbB <- xgboost::xgboost(
    data=x9_trainB_bal_dmat,
    nrounds=x9_xgb$niter,    # <-- safer in case we add more rounds manually
    params = x9_params,
    print_every_n=1, save_period = NULL, save_name = NULL)


# train model A first
x9_xgbA <- xgboost::xgboost(
    data=x9_trainA_bal_dmat,
    nrounds=x9_xgb$niter,
    params=x9_params,
    print_every_n=1, save_period = NULL, save_name = NULL)

# write it all out    
saveRDS(x9_xgbA, file=paste0(fp_ft_modsA, "ft_x9_", x9_target, "_mod.rds"))
saveRDS(x9_xgbB, file=paste0(fp_ft_modsB, "ft_x9_", x9_target, "_mod.rds"))
saveRDS(x9_xgb, file=paste0(fp_ft_modsAB, "ft_x9_", x9_target, "_mod.rds"))
saveRDS(names(x9_train_bal), file=paste0(fp_ft_feats, "ft_x9_", x9_target, "_mod.rds"))

# update remaining features
x9_rem_feats <- readRDS(fp_ft_remaining_feats)
x9_rem_feats <- setdiff(x9_rem_feats, x9_target)
saveRDS(x9_rem_feats, fp_ft_remaining_feats)

rm(list=ls()[grepl("^x9_", ls())])
gc()



# x10 --------------------------
# target: ps_ind_04_cat
# type: 8 class multi
# removals: target, id, 
set.seed(this_seed)
x10_rem_feats <- readRDS(fp_ft_remaining_feats)
x10_target <- "ps_ind_10_bin"
table(train[, x10_target])
x10_samp_size <- 1500


# classes should start at zero
x10_trainA <- data.frame(trainA)
x10_trainB <- data.frame(trainB)


# do this before fixing target variable
x10_ho_indx <- caret::createDataPartition(y=trainA[, x10_target], times=1, p=0.10, list=F)
x10_ho <- rbind(x10_trainA[x10_ho_indx, ], x10_trainB[x10_ho_indx, ])
x10_trainA <- x10_trainA[-x10_ho_indx, ]
x10_trainB <- x10_trainB[-x10_ho_indx, ]

# fix target value here if necessary
x10_trainA <- x10_trainA[x10_trainA[, x10_target] >= 0,]
x10_trainB <- x10_trainB[x10_trainB[, x10_target] >= 0, ]
table(x10_trainA[, x10_target])
table(x10_trainB[, x10_target])


# unique target values should be the same across A/B
assert_that(all(intersect(unique(x10_trainA[, x10_target]), unique(x10_trainB[, x10_target])) %in% x10_trainA[, x10_target]))
assert_that(all(intersect(unique(x10_trainA[, x10_target]), unique(x10_trainB[, x10_target])) %in% x10_trainB[, x10_target]))

# take equalized samples of each of the target variable
x10_trainA_bal <- take_equal_sample(x10_trainA, x10_samp_size, x10_target, unique(x10_trainA[, x10_target]))
x10_trainB_bal <- take_equal_sample(x10_trainB, x10_samp_size, x10_target, unique(x10_trainB[, x10_target]))
x10_train_bal <- rbind(x10_trainA_bal, x10_trainB_bal)    
table(x10_trainA_bal[, x10_target])


x10_trainA_bal_y <- x10_trainA_bal[, x10_target]
x10_trainB_bal_y <- x10_trainB_bal[, x10_target]
x10_train_bal_y <- c(x10_trainA_bal_y, x10_trainB_bal_y)
x10_ho_y <- x10_ho[, x10_target]
x10_removals <- c(x10_target, "target", "id", "ps_ind_14") #, "ps_ind_07_bin", "ps_ind_08_bin", "ps_ind_06_bin")


x10_trainA_bal <- x10_trainA_bal[, setdiff(names(x10_trainA_bal), x10_removals)]
x10_trainB_bal <- x10_trainB_bal[, setdiff(names(x10_trainB_bal), x10_removals)]
x10_train_bal <- x10_train_bal[, setdiff(names(x10_train_bal), x10_removals)]
x10_ho <- x10_ho[, setdiff(names(x10_ho), x10_removals)]

x10_trainA_bal_dmat <- xgb.DMatrix(as.matrix(x10_trainA_bal), label=x10_trainA_bal_y)
x10_trainB_bal_dmat <- xgb.DMatrix(as.matrix(x10_trainB_bal), label=x10_trainB_bal_y)
x10_train_bal_dmat <- xgb.DMatrix(as.matrix(x10_train_bal), label=x10_train_bal_y)
x10_ho_dmat <- xgb.DMatrix(as.matrix(x10_ho), label=x10_ho_y)

x10_params <-  list(
    "objective" = "binary:logistic",
    "eval_metric" = "auc",  
    # "objective" = "multi:softprob",
    # "eval_metric" = "mlogloss",
    # "num_class" = length(unique(x10_trainA_bal_y)),  # <-- 8 classes including "0"
    "eta" = 0.4,
    "max_depth" = 5,
    "subsample" = 0.8,
    "colsample_bytree" = 0.5,
    "lambda" = 0,
    "alpha" = 1,
    "gamma" = 0,
    "max_delta_step" = 1,
    "scale_pos_weight" = 1,
    "nthread" = 4) # 131 best nrounds


# first train the combined model
x10_nrounds <- 5
x10_xgb <- xgboost::xgboost(
    nrounds=x10_nrounds,
    # nrounds=1, xgb_model=x10_xgb,  # for additional training rounds on already trained model
    data=x10_train_bal_dmat,
    params=x10_params,
    print_every_n=1, save_period = NULL, save_name = NULL)


{
    # feature importance for cheater removal
    x10_feat_imp <- xgboost::xgb.importance(feature_names = names(x10_train_bal), model=x10_xgb)
    xgboost::xgb.plot.importance(x10_feat_imp[1:20,])
    
    
    # control flow for which plotting route to take based on the type of model 
    if(x10_params$objective == "multi:softprob") {
        # multi classification
        x10_preds <- as.data.frame(matrix(predict(x10_xgb, x10_ho_dmat), ncol=length(unique(x10_train_bal_y)), byrow=T))
        names(x10_preds) <- paste0("feat_tar_x10_", x10_target, "_", sprintf("%02.0f", 0:(length(unique(x10_trainA_bal_y)) - 1)))
        x10_preds <- cbind(x10_ho_y, x10_preds)
        x10_preds_gath <- tidyr::gather(x10_preds, pred_cat, pred_val, -x10_ho_y) %>% 
            group_by(x10_ho_y, pred_cat) %>%
            summarise(mean_pred_val = mean(pred_val, na.rm=T)) %>% ungroup
        ggplot(x10_preds_gath, aes(x=x10_ho_y, y=pred_cat, fill=mean_pred_val)) +
            geom_tile(color="White", size=0.1) +
            # scale_fill_viridis(name="Mean Prediction") +
            scale_fill_viridis(name="Mean Prediction", limits=c(0.001, (2* (1/length(unique(x10_train_bal_y)))))) +
            coord_equal() +  
            ggtitle(paste0("Mean Predictions: ", x10_target, " after ", x10_xgb$niter, " nrounds"))
    } else if(x10_params$objective == "binary:logistic") {
        # binary classification -- not tested yet
        x10_preds <- data.frame(actual=x10_ho_y, preds=predict(x10_xgb, x10_ho_dmat))
        ggplot(x10_preds, aes(x=preds, fill=as.factor(actual))) +
            geom_density(alpha=0.5, position='identity') +
            ggtitle(paste0("Prediction density: ", x10_target, " after ", x10_xgb$niter, " nrounds"))
    } else if(x10_params$objective == "reg:linear") {
        # linear regression -- not tested yet
        x10_preds <- data.frame(actual=x10_ho_y, preds=predict(x10_xgb, x10_ho_dmat))
        x10_preds$residual <- x10_preds$actual - x10_preds$preds
        ggplot(x10_preds, x=actual, y=residual) +
            geom_point(alpha=0.4) + 
            ggtitle(paste0("Prediction residuals: ", x10_target, " after ", x10_xgb$niter, " nrounds"))
    }
}  # feature importance and model plot


# in depth analysis of multi classi
x10_preds_manual_gath <- tidyr::gather(x10_preds, pred_cat, pred_val, -x10_ho_y)
x10_ho_y_val <- 3
ggplot(data=x10_preds_manual_gath %>% filter(x10_ho_y == x10_ho_y_val), aes(x=pred_val, fill=as.factor(pred_cat))) +
    geom_histogram(alpha=0.3, position='identity') +
    ggtitle(paste0("Investigating where ho_y = ", x10_ho_y_val, " and nrounds = ", x10_xgb$niter))


# if results above look good, train model B and the combined AB model
x10_xgbB <- xgboost::xgboost(
    data=x10_trainB_bal_dmat,
    nrounds=x10_xgb$niter,    # <-- safer in case we add more rounds manually
    params = x10_params,
    print_every_n=1, save_period = NULL, save_name = NULL)


# train model A first
x10_xgbA <- xgboost::xgboost(
    data=x10_trainA_bal_dmat,
    nrounds=x10_xgb$niter,
    params=x10_params,
    print_every_n=1, save_period = NULL, save_name = NULL)

# write it all out    
saveRDS(x10_xgbA, file=paste0(fp_ft_modsA, "ft_x10_", x10_target, "_mod.rds"))
saveRDS(x10_xgbB, file=paste0(fp_ft_modsB, "ft_x10_", x10_target, "_mod.rds"))
saveRDS(x10_xgb, file=paste0(fp_ft_modsAB, "ft_x10_", x10_target, "_mod.rds"))
saveRDS(names(x10_train_bal), file=paste0(fp_ft_feats, "ft_x10_", x10_target, "_mod.rds"))

# update remaining features
x10_rem_feats <- readRDS(fp_ft_remaining_feats)
x10_rem_feats <- setdiff(x10_rem_feats, x10_target)
saveRDS(x10_rem_feats, fp_ft_remaining_feats)

rm(list=ls()[grepl("^x10_", ls())])
gc()



# x11 --------------------------
# target: ps_ind_04_cat
# type: 8 class multi
# removals: target, id, 
set.seed(this_seed)
x11_rem_feats <- readRDS(fp_ft_remaining_feats)
x11_target <- "ps_ind_10_bin"
table(train[, x11_target])
x11_samp_size <- 1500


# classes should start at zero
x11_trainA <- data.frame(trainA)
x11_trainB <- data.frame(trainB)


# do this before fixing target variable
x11_ho_indx <- caret::createDataPartition(y=trainA[, x11_target], times=1, p=0.10, list=F)
x11_ho <- rbind(x11_trainA[x11_ho_indx, ], x11_trainB[x11_ho_indx, ])
x11_trainA <- x11_trainA[-x11_ho_indx, ]
x11_trainB <- x11_trainB[-x11_ho_indx, ]

# fix target value here if necessary
x11_trainA <- x11_trainA[x11_trainA[, x11_target] >= 0,]
x11_trainB <- x11_trainB[x11_trainB[, x11_target] >= 0, ]
table(x11_trainA[, x11_target])
table(x11_trainB[, x11_target])


# unique target values should be the same across A/B
assert_that(all(intersect(unique(x11_trainA[, x11_target]), unique(x11_trainB[, x11_target])) %in% x11_trainA[, x11_target]))
assert_that(all(intersect(unique(x11_trainA[, x11_target]), unique(x11_trainB[, x11_target])) %in% x11_trainB[, x11_target]))

# take equalized samples of each of the target variable
x11_trainA_bal <- take_equal_sample(x11_trainA, x11_samp_size, x11_target, unique(x11_trainA[, x11_target]))
x11_trainB_bal <- take_equal_sample(x11_trainB, x11_samp_size, x11_target, unique(x11_trainB[, x11_target]))
x11_train_bal <- rbind(x11_trainA_bal, x11_trainB_bal)    
table(x11_trainA_bal[, x11_target])


x11_trainA_bal_y <- x11_trainA_bal[, x11_target]
x11_trainB_bal_y <- x11_trainB_bal[, x11_target]
x11_train_bal_y <- c(x11_trainA_bal_y, x11_trainB_bal_y)
x11_ho_y <- x11_ho[, x11_target]
x11_removals <- c(x11_target, "target", "id", "ps_ind_14") #, "ps_ind_07_bin", "ps_ind_08_bin", "ps_ind_06_bin")


x11_trainA_bal <- x11_trainA_bal[, setdiff(names(x11_trainA_bal), x11_removals)]
x11_trainB_bal <- x11_trainB_bal[, setdiff(names(x11_trainB_bal), x11_removals)]
x11_train_bal <- x11_train_bal[, setdiff(names(x11_train_bal), x11_removals)]
x11_ho <- x11_ho[, setdiff(names(x11_ho), x11_removals)]

x11_trainA_bal_dmat <- xgb.DMatrix(as.matrix(x11_trainA_bal), label=x11_trainA_bal_y)
x11_trainB_bal_dmat <- xgb.DMatrix(as.matrix(x11_trainB_bal), label=x11_trainB_bal_y)
x11_train_bal_dmat <- xgb.DMatrix(as.matrix(x11_train_bal), label=x11_train_bal_y)
x11_ho_dmat <- xgb.DMatrix(as.matrix(x11_ho), label=x11_ho_y)

x11_params <-  list(
    "objective" = "binary:logistic",
    "eval_metric" = "auc",  
    # "objective" = "multi:softprob",
    # "eval_metric" = "mlogloss",
    # "num_class" = length(unique(x11_trainA_bal_y)),  # <-- 8 classes including "0"
    "eta" = 0.4,
    "max_depth" = 5,
    "subsample" = 0.8,
    "colsample_bytree" = 0.5,
    "lambda" = 0,
    "alpha" = 1,
    "gamma" = 0,
    "max_delta_step" = 1,
    "scale_pos_weight" = 1,
    "nthread" = 4) # 131 best nrounds


# first train the combined model
x11_nrounds <- 5
x11_xgb <- xgboost::xgboost(
    nrounds=x11_nrounds,
    # nrounds=1, xgb_model=x11_xgb,  # for additional training rounds on already trained model
    data=x11_train_bal_dmat,
    params=x11_params,
    print_every_n=1, save_period = NULL, save_name = NULL)


{
    # feature importance for cheater removal
    x11_feat_imp <- xgboost::xgb.importance(feature_names = names(x11_train_bal), model=x11_xgb)
    xgboost::xgb.plot.importance(x11_feat_imp[1:20,])
    
    
    # control flow for which plotting route to take based on the type of model 
    if(x11_params$objective == "multi:softprob") {
        # multi classification
        x11_preds <- as.data.frame(matrix(predict(x11_xgb, x11_ho_dmat), ncol=length(unique(x11_train_bal_y)), byrow=T))
        names(x11_preds) <- paste0("feat_tar_x11_", x11_target, "_", sprintf("%02.0f", 0:(length(unique(x11_trainA_bal_y)) - 1)))
        x11_preds <- cbind(x11_ho_y, x11_preds)
        x11_preds_gath <- tidyr::gather(x11_preds, pred_cat, pred_val, -x11_ho_y) %>% 
            group_by(x11_ho_y, pred_cat) %>%
            summarise(mean_pred_val = mean(pred_val, na.rm=T)) %>% ungroup
        ggplot(x11_preds_gath, aes(x=x11_ho_y, y=pred_cat, fill=mean_pred_val)) +
            geom_tile(color="White", size=0.1) +
            # scale_fill_viridis(name="Mean Prediction") +
            scale_fill_viridis(name="Mean Prediction", limits=c(0.001, (2* (1/length(unique(x11_train_bal_y)))))) +
            coord_equal() +  
            ggtitle(paste0("Mean Predictions: ", x11_target, " after ", x11_xgb$niter, " nrounds"))
    } else if(x11_params$objective == "binary:logistic") {
        # binary classification -- not tested yet
        x11_preds <- data.frame(actual=x11_ho_y, preds=predict(x11_xgb, x11_ho_dmat))
        ggplot(x11_preds, aes(x=preds, fill=as.factor(actual))) +
            geom_density(alpha=0.5, position='identity') +
            ggtitle(paste0("Prediction density: ", x11_target, " after ", x11_xgb$niter, " nrounds"))
    } else if(x11_params$objective == "reg:linear") {
        # linear regression -- not tested yet
        x11_preds <- data.frame(actual=x11_ho_y, preds=predict(x11_xgb, x11_ho_dmat))
        x11_preds$residual <- x11_preds$actual - x11_preds$preds
        ggplot(x11_preds, x=actual, y=residual) +
            geom_point(alpha=0.4) + 
            ggtitle(paste0("Prediction residuals: ", x11_target, " after ", x11_xgb$niter, " nrounds"))
    }
}  # feature importance and model plot


# in depth analysis of multi classi
x11_preds_manual_gath <- tidyr::gather(x11_preds, pred_cat, pred_val, -x11_ho_y)
x11_ho_y_val <- 3
ggplot(data=x11_preds_manual_gath %>% filter(x11_ho_y == x11_ho_y_val), aes(x=pred_val, fill=as.factor(pred_cat))) +
    geom_histogram(alpha=0.3, position='identity') +
    ggtitle(paste0("Investigating where ho_y = ", x11_ho_y_val, " and nrounds = ", x11_xgb$niter))


# if results above look good, train model B and the combined AB model
x11_xgbB <- xgboost::xgboost(
    data=x11_trainB_bal_dmat,
    nrounds=x11_xgb$niter,    # <-- safer in case we add more rounds manually
    params = x11_params,
    print_every_n=1, save_period = NULL, save_name = NULL)


# train model A first
x11_xgbA <- xgboost::xgboost(
    data=x11_trainA_bal_dmat,
    nrounds=x11_xgb$niter,
    params=x11_params,
    print_every_n=1, save_period = NULL, save_name = NULL)

# write it all out    
saveRDS(x11_xgbA, file=paste0(fp_ft_modsA, "ft_x11_", x11_target, "_mod.rds"))
saveRDS(x11_xgbB, file=paste0(fp_ft_modsB, "ft_x11_", x11_target, "_mod.rds"))
saveRDS(x11_xgb, file=paste0(fp_ft_modsAB, "ft_x11_", x11_target, "_mod.rds"))
saveRDS(names(x11_train_bal), file=paste0(fp_ft_feats, "ft_x11_", x11_target, "_mod.rds"))

# update remaining features
x11_rem_feats <- readRDS(fp_ft_remaining_feats)
x11_rem_feats <- setdiff(x11_rem_feats, x11_target)
saveRDS(x11_rem_feats, fp_ft_remaining_feats)

rm(list=ls()[grepl("^x11_", ls())])
gc()






