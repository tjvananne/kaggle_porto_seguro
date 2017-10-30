

# feat targeting config file

library(dplyr)
library(caret)
library(xgboost)
library(tidyr)
library(ggplot2)
library(viridis)
library(assertthat)




take_equal_sample <- function(p_df, p_sampsize, p_target, p_unq_tars) {
    
    # i want to explicitly supply unique target values in case I want to leave any out
    # some, for example, are too sparse (less than 10 instances of that value present in training data)
    
    og_names <- names(p_df)
    result_mat <- matrix(rep(rep(0, p_sampsize * length(p_unq_tars)), ncol(p_df)), ncol=ncol(p_df))
    indx_start <- 1
    indx_end <- p_sampsize
    for(i in 1:length(p_unq_tars)) {
        # browser()
        print(i)
        p_df_filtered_ <- p_df[p_df[, p_target] == p_unq_tars[i],    ]      # filter to desired value
        replace_if_ <- nrow(p_df_filtered_) < p_sampsize                     # sample with replacement if target values less than samp size
        result_mat[indx_start:indx_end,] <- as.matrix(p_df_filtered_[sample(x=1:nrow(p_df_filtered_), size=p_sampsize, replace=replace_if_), ])  
        
        # shift matrix indeces
        indx_start <- indx_start + p_sampsize
        indx_end <- indx_end + p_sampsize
        
    }
    result_df <- as.data.frame(result_mat)
    names(result_df) <- og_names
    return(result_df)
}

# example
# res_mat <- take_equal_sample(x1_trainA, x1_samp_size, x1_target, unique(x1_trainA[, x1_target]))
