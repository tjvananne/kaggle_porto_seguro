



# Processing level 1 models:

# there are a few unique things I want to accomplish in this script. Ideally, I think I should
# split this into at least a few different scripts:
#    1) load in prior train predictions for training intermediate level2 stacks 
#    2) load in prior train predictions for a final level 2 stacked model
#    2.5) pass HO/test through a single lvl1 model to measure local/leaderboard score
#    3) run HO through lvl 1 models to gather predictions
#    3.5) run HO through lvl 2 models to gather predictions 
#    4) run test through lvl 1 models to gather predictions
#    4.5) run test through lvl 2 models to gather predictions



source("r_scripts/lvl1_xgb_config.R")
load("cache/level1_files.RData")
rm(train, trainA, trainB, Y) # I just want idA, idB, idHO, YA, YB, YHO 
test <- readRDS("input/test.rds")
samp <- read.csv("input/sample_submission.csv", stringsAsFactors = F)
lvl1_results <- read.csv("cache/level1_results.csv", stringsAsFactors = F)
gc()


# single model holdout gini / PLB gini comparison -----------------------------------------------


#' lvl1_results %>% filter(eval_metric == "logloss") %>%  # "auc" is the other option
#'     arrange(cv_score) %>% 
#'     select(-paramkey) %>%
#'     top_n(10, -cv_score)
#' 
#' # let's use level 1 and iteration 0036 for first gini / plb gini comparison
#' single_mod <- readRDS(list.files(fp_dir_models, full.names = T)[grepl("0036", list.files(fp_dir_models))])
#' single_feats <- readRDS(list.files(fp_dir_feats, full.names = T)[grepl("0036", list.files(fp_dir_models))])
#' single_HO <- trainHO %>% select(single_feats)
#' single_HO_dmat <- xgb.DMatrix(as.matrix(single_HO))
#' single_HO_preds <- predict(single_mod, single_HO_dmat)
#' normalizedGini(YHO, single_HO_preds)
#' 
#' 
#' single_test <- test %>% select(single_feats)
#' single_test_dmat <- xgb.DMatrix(as.matrix(single_test))
#' single_test_preds <- predict(single_mod, single_test_dmat)
#' single_test_sub <- data.frame(id=test$id, target=single_test_preds)
#' write.csv(single_test_sub, "subs/06_0036_0dot268_ho_gini_single_best_logloss.csv", row.names = F)
#' 
#'     #' 0036 logloss model scored 0.268 gini for local holdout and 0.269 on public leader board








# loading and process training preds (for level 2 FINAL stacker - will need another version for intermediate stacker) ---------

# identify necessary files
stack1_preds <- list.files("cache/level1_preds", full.names = T)
stack1_feats <- list.files("cache/level1_feats", full.names = T)
stack1_models <- list.files("cache/level1_models", full.names = T)


# initialize space, loop through models to capture predictions, one model at a time
stack1_preds_mat <- matrix(rep(rep(0, (length(YA) + length(YB))), length(stack1_preds)), ncol=length(stack1_preds))
for(i in 1:length(stack1_preds)) {
    stack1_preds_ <- readRDS(stack1_preds[i])
    if(i == 1) {
        ids_ <- stack1_preds_$id  # store ids on first iteration, assert that id's on all files loaded in match
    } else {
        assert_that(all(stack1_preds_$id == ids_))
    }
    stack1_preds_mat[, i] <- stack1_preds_[, 2]
}


# training an elastic net model
glmn_cv1 <- cv.glmnet(x=stack1_preds_mat, y=c(YA, YB), family="binomial")
plot(glmn_cv1)
glmn_cv1$lambda.min
glmn_cv1$lambda.1se



# training another xgbooster 
params1 = list(
    "objective" = "binary:logistic",
    "eval_metric" = "auc",  
    "eta" = 0.01,
    "max_depth" = 7,
    "subsample" = 0.8,
    "colsample_bytree" = 0.5,
    "lambda" = 0,
    "alpha" = 1,
    "gamma" = 0,
    "max_delta_step" = 1,
    "scale_pos_weight" = 10,
    "nthread" = 4)

set.seed(1776)
xgbcv_lvl2_1 <- xgb.cv(
    params = params1,
    nfold=5,
    data = xgb.DMatrix(stack1_preds_mat, label=c(YA, YB)),
    nrounds=10000,
    early_stopping_rounds = 100)  # 100 is probably overkill here...


xgb_lvl2_1 <- xgb.train(
    params = params1,
    data = xgb.DMatrix(stack1_preds_mat, label=c(YA, YB)),
    nrounds = 165)








# now pass through all of the holdout records through these models
stack1_HO_preds_mat <- matrix(  rep(rep(0, nrow(trainHO)), length(stack1_models)), ncol = length(stack1_models))
for(i in 1:length(stack1_models)) {
    feats_ <- readRDS(stack1_feats[i])
    trainHO_ <- trainHO[, feats_]
    model_ <- readRDS(stack1_models[i])
    stack1_HO_preds_mat[,i] <- predict(model_, xgb.DMatrix(as.matrix(trainHO_)))
}

lvl2_HO_stack <- as.numeric(predict(glmn_cv1, newx=stack1_HO_preds_mat, s="lambda.min"))
normalizedGini(YHO, lvl2_HO_stack)

lvl2_HO_stack_maxreg <- as.numeric(predict(glmn_cv1, newx=stack1_HO_preds_mat, s="lambda.1se"))
normalizedGini(YHO, lvl2_HO_stack_maxreg)

xgb_lvl2_1_preds <- predict(xgb_lvl2_1, xgb.DMatrix(stack1_HO_preds_mat))
normalizedGini(YHO, xgb_lvl2_1_preds)  # 0.2769



# generating level1 test predictions --------------------------------------------------------------


# initialize a zero'd-out matrix to gather predictions
test_pred_mat <- matrix(rep(rep(0, nrow(test)), length(stack1_preds)), nrow=nrow(test))
for(i in 1:length(stack1_preds)) {
    
    # read in feats used for this model and subset test by those
    feats_ <- readRDS(stack1_feats[i])
    test_dmat_ <- xgb.DMatrix(as.matrix(test[, feats_]))
    
    # read in model and pass test features into it for predictions
    xgb_mod_ <- readRDS(stack1_models[i])
    test_pred_mat[, i] <- predict(xgb_mod_, test_dmat_)
        
}


lvl2_test_stack <- as.numeric(predict(glmn_cv1, newx=test_pred_mat, s="lambda.min", type="response"))
sub_glm <- data.frame(id=test$id, target=lvl2_test_stack)
write.csv(sub, "subs/07_118_lvl1_mods_combined_with_glmnet_276_gini.csv", row.names = F)

xgb_lvl2_1_subpreds <- predict(xgb_lvl2_1, xgb.DMatrix(test_pred_mat))
sub_xgb1 <- data.frame(id=test$id, target=xgb_lvl2_1_subpreds)
write.csv(sub, "subs/08_118_lvl1_mods_combined_with_xgb_auc_277_gini.csv", row.names = F)


sub_bagged_glm_xgb <- data.frame(id=test$id, target=  ((lvl2_test_stack + xgb_lvl2_1_subpreds) / 2))
write.csv(sub_bagged_glm_xgb, "subs/09_118_lvl1_mods_combined_with_glmnet_xgb_bagged.csv", row.names = F)



# bag all (mean) the values across rows (different models) to get mean score
test_pred_mat_bagged <- apply(test_pred_mat, 1, mean)
bagged_sub <- data.frame(id=test$id, target=test_pred_mat_bagged)
write.csv(bagged_sub, "subs/05_bagged_sub_all_after_94_iters.csv", row.names=F)



# correlation matrix
col <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))
test_pred_cor <- cor(test_pred_mat)
corrplot(test_pred_cor, method=c("ellipse"), type="lower")
corrplot(test_pred_cor, method=c("number"), type="lower")

hist(test_pred_mat[, 5], col='light blue')




# submission format
sub <- data.frame(id=samp$id, target=test_pred_mat_bagged)


# can also feed these into a second model 


