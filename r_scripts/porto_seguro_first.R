

# first script 

library(dplyr)
library(caret)
library(Matrix)


list.files('input')
#samp <- read.csv("input/sample_submission.csv", stringsAsFactors = F)
#train <- read.csv("input/train.csv", stringsAsFactors = F)
#test <- read.csv("input/test.csv", stringsAsFactors = F)
#saveRDS(train, "input/train.rds")
#saveRDS(test, "input/test.rds")
train <- readRDS("input/train.rds")
test <- readRDS("input/test.rds")


    # sapply(train, function(x) sum(is.na(x)))
    # sapply(train, function(x) round(sum(x == 0) / nrow(train) * 100, digits = 2))
    # sapply(train, function(x) length(unique(x)))
    # table(sapply(train, function(x) length(unique(x)))) %>% data.frame() %>% arrange(desc(Var1))
    # table(sapply(test, function(x) length(unique(x)))) %>% data.frame() %>% arrange(desc(Var1))


# isolate features
train_feats <- train %>% select(-id, -target)
test_feats <- test %>% select(-id)

    # # x1 <- as.matrix(test_feats[1:450000, ])
    # x1 <- as.matrix(test_feats[450001:nrow(test_feats), ])
    # num_rows <- nrow(x1)
    # num_cols <- ncol(x1)
    # 
    # # begin 2-way interaction data
    # x_sub <- gtools::combinations(num_cols, 2)
    # x_result <- matrix(rep(rep(0, nrow(x_sub)), num_rows), nrow=num_rows)
    # 
    # start <- Sys.time()
    # for(i in 1:nrow(x_sub)) {
    #     x1_ <- x1[, x_sub[i,]]
    #     x1_mult_ <- matrix(x1_[, 1] * x1_[, 2], ncol=1)
    #     x_result[, i] <- x1_mult_
    #     # if(i %% 50 == 0) {
    #     #     print("garbage collecting: ")
    #     #     gc()
    #     # }
    # }
    # Sys.time() - start
    # 
    # saveRDS(x_result, "cache/test_2nd_half_feats_raw_2way_dense_mat.rds")


    # # create 2-way interactions and store in sparse matrix
    # train_feats_interaction <- sparse.model.matrix(~ . ^2, data=train_feats)
    # saveRDS(train_feats_interaction, "input/train_feats_2w_interaction.rds")
    # test_feats_interaction <- sparse.model.matrix(~ . ^2, data=test_feats)
    # saveRDS(test_feats_interaction, "input/test_feats_2w_interaction.rds")




ppScale <- caret::preProcess(train_feats, method=c("center", "scale"))
train_center_scale <- cbind(data.frame(id=train$id, target=train$target, stringsAsFactors = F),  predict(ppScale, train_feats))
test_center_scale <- cbind(data.fr)




# split cols into:
#    - binary categorical
#    - multi-categorical (3 - 26 categories looks like good heuristic)
#    - continuous        (more than 26 categories)

table(train$target)


list.files('input')

saveRDS(train, "input/train.rds")
saveRDS(test, "input/test.rds")






