

source("r_scripts/lvl1_xgb_config.R")



    # directory setup:
    if(!dir.exists(fp_dir_preds)) {
        dir.create(fp_dir_preds)
    }

    if(!dir.exists(fp_dir_models)) {
        dir.create(fp_dir_models)
    }
    
    if(!dir.exists(fp_dir_feats)) {
        dir.create(fp_dir_feats)
    }
    
    if(!file.exists(fp_results)) {
        cat("level,iteration,paramkey,cv_score,best_nrounds\n", file=fp_results)
    }



    # experiment data setup
    if(!file.exists(data_cache_fp)) {
        print("cache didn't exists, creating level 1 data split now...")
        
        train <- readRDS('input/train.rds')
            assert_that(all(train$id[order(train$id)] == train$id))
        
        set.seed(1776)
        indxA <- caret::createDataPartition(train$target, p=0.5, list=F)
        trainA <- train[indxA, ]
        YA <- train$target[indxA]
        idA <- train$id[indxA]
        trainA$target <- NULL
        
        trainB <- train[-indxA, ]
        YB <- train$target[-indxA]
        idB <- train$id[-indxA]
        trainB$target <- NULL 
        
        Y <- train$target
        train$target <- NULL
        
        save(train, Y, trainA, YA, idA, trainB, YB, idB, file=data_cache_fp)
    } else {
        print("cache exists, loading level 1 data split now...")
        load(data_cache_fp)
    }



set.seed(1776)

all_xgb_params <- list(
    "objective" = "binary:logistic",
    "eval_metric" = c("logloss", "error", "auc"),  # <- must handle min/max depending on eval metric
    "eta" = c(0.01, 0.05, 0.1),
    "max_depth" = c(3, 5, 7),
    "subsample" = c(0.4, 0.6, 0.9),
    "colsample_bytree" = c(0.5, 0.7, 0.9),
    "lambda" = c(0, 1),
    "alpha" = c(0, 1),
    "gamma" = c(0, 2, 4),
    "max_delta_step" = c(0, 1),
    "scale_pos_weight" = c(1, 10, 20),
    "nthread" = 4)

xgb_paramgrid <- expand.grid(all_xgb_params, stringsAsFactors = F)

col_names <- names(train)[!grepl("^id$", names(train))]



# attempt to pick up where we left off if possible:
prior_results <- read.csv(file=fp_results, stringsAsFactors = F)
if(length(prior_results$iteration) > 0) {
    min_i <- (max(prior_results$iteration) + 1)  # <-- pick up where we left off
} else {
    min_i <- 1  # <-- start from scratch for this level
}



for(i in min_i:999) {
    print(paste0("***** at iteration **********************************************  ", i))
    set.seed((1776 * i))
    
    # set up file paths:
    iter <- sprintf("%03.0f", i)
    fp_model <- file.path(fp_dir_models, paste0("lvl_", exp_level, "_model_", iter, ".rds"))
    fp_feats <- file.path(fp_dir_feats, paste0("lvl_", exp_level, "_feats_", iter, ".rds"))
    fp_preds <- file.path(fp_dir_preds, paste0("lvl_", exp_level, "_feats_", iter, ".rds"))
    
    
    # select xgb params:
    xgb_params <- as.list(xgb_paramgrid[sample.int(nrow(xgb_paramgrid), size=1), ])
    xgb_paramkey <- paste0(paste0(names(xgb_params), "="), xgb_params, collapse="---")
    
    
    
    # select features for this iteration:
    num_col_sample <- floor(runif(1, (.15 * ncol(train)),  (.7 * ncol(train))   ))
    feat_sample <- col_names[sample.int(length(col_names), size=num_col_sample)]
        
    train_ <- train[, feat_sample]
    trainA_ <- trainA[, feat_sample]
    trainB_ <- trainB[, feat_sample]
    
    train_dmat_ <- xgb.DMatrix(as.matrix(train_), label=Y)
    trainA_dmat_ <- xgb.DMatrix(as.matrix(trainA_), label=YA)
    trainB_dmat_ <- xgb.DMatrix(as.matrix(trainB_), label=YB)            
    
    
    
    # find best nrounds
    print("training cross validation...")
    xgb_cv_ <- xgb.cv(
        data=train_dmat_,
        nfold=5,
        params=xgb_params,
        nrounds=1500,
        early_stopping_rounds=40,
        print_every_n=1501)
    
    
    eval_log_ <- data.frame(xgb_cv_$evaluation_log, stringsAsFactors = F)
    eval_metric_ <- names(eval_log_)[grepl("^test_", names(eval_log_)) & grepl("_mean$", names(eval_log_))]
    if(xgb_params$eval_metric == "auc") {
        # we want to maximize auc
        best_nrounds <- which.max(eval_log_[, eval_metric_])
        cv_score <- max(eval_log_[, eval_metric_])
    } else {
        # we want to minimize error and logloss
        best_nrounds <- which.min(eval_log_[, eval_metric_])
        cv_score <- min(eval_log_[, eval_metric_])
    }
    
    
    # train on all data
    print("training model on all of train...")
    xgb_all_ <- xgboost(
        data=train_dmat_,
        nrounds=best_nrounds,
        params=xgb_params,
        print_every_n = 1501,
        save_period = NULL)
    saveRDS(xgb_all_, fp_model)
    
    
    # train on A, predict on B
    print("training model on trainA...")
    xgbA_ <- xgboost(
        data=trainA_dmat_,
        nrounds=best_nrounds,
        params=xgb_params,
        print_every_n=1501,
        save_period = NULL)
    predsB_ <- predict(xgbA_, trainB_dmat_)
    
    
    # train on B, predict on A
    print("training model on trainB...")
    xgbB_ <- xgboost(
        data=trainB_dmat_,
        nrounds=best_nrounds,
        params=xgb_params,
        print_every_n=1501,
        save_period=NULL)
    predsA_ <- predict(xgbB_, trainA_dmat_)
    
    
    print("generating predictions and saving to disk...")
    preds_ <- data.frame(id=c(idA, idB), colname = c(predsA_, predsB_))
    names(preds_) <- c("id", paste0("preds_", exp_level, "_", iter))
    
    
    # write out predictions
    saveRDS(preds_, file=fp_preds)
    
    # write out features
    saveRDS(feat_sample, file=fp_feats)
    
    # write out results
    cat( paste0(exp_level, ",", iter, ",", xgb_paramkey, ",", cv_score, ",", best_nrounds, "\n"),  file=fp_results, append=T)
    gc()
}






