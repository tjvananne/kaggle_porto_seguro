

# first script 

train <- read.csv("input/train.csv", stringsAsFactors = F)
test <- read.csv("input/test.csv", stringsAsFactors = F)

saveRDS(train, "input/train.rds")
saveRDS(test, "input/test.rds")

sapply(train, function(x) sum(is.na(x)))
