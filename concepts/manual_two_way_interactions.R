

library(gtools)
library(ggplot2)


num_rows <- 500000
num_cols <- 50



# matrix of numeric features
x1 <- matrix(runif(num_rows * num_cols, 1, 10), nrow=num_rows)
# x1 <- matrix(1:(num_rows*num_cols), nrow=num_rows)
x1


# combinations of those feature columns
x_sub <- gtools::combinations(num_cols, 2)
x_sub



# x_result <- matrix(rep(0, nrow(x_sub) * num_rows), nrow=num_rows)  # this works, except throws integer overflow
x_result <- matrix(rep(rep(0, nrow(x_sub)), num_rows), nrow=num_rows)
x_result

dim(x1); dim(x_sub); dim(x_result)



start <- Sys.time()
for(i in 1:nrow(x_sub)) {
    x1_ <- x1[, x_sub[i,]]
    x1_mult_ <- matrix(x1_[, 1] * x1_[, 2], ncol=1)
    x_result[, i] <- x1_mult_
}
Sys.time() - start




assert_that( all((x1[, 1] * x1[,2]) == x_result[, 1]))
assert_that( all((x1[, 1] * x1[,3]) == x_result[, 2]))



# 500000 x 50,  23.7 seconds
# 50000  x 50,  2.4  seconds
# 5000   x 50,  0.16 seconds
# 500    x 50,  0.03 seconds
# 50     x 50,  0.01 seconds


profile <- data.frame(
    rows=c(50, 500, 5000, 50000, 500000),
    seconds=c(0.01, 0.03, 0.16, 2.4, 23.7)
)


# as linear of a profile as it gets...
ggplot(data=profile, aes(x=rows, y=seconds)) +
    geom_line(size=2)


profile$ratio <- profile$rows / profile$seconds



