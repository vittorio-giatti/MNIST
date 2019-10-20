setwd("C:/Users/Vittorio/Desktop/MNIST")
source("lift-roc1.R")
library(readr)
train <- read_csv("C:/Users/Vittorio/Desktop/MNIST/train.csv")
train=data.frame(train)
train[,1]=factor(train[,1])

test <- read_csv("C:/Users/Vittorio/Desktop/MNIST/test.csv")
test=data.frame(test)

test_matrix=test
library("xgboost")

#train_matrix <- xgb.DMatrix(data = train[1:20,-1],label = train[1:20,1])

train$label <- as.numeric(train$label)

# Make split index
#train_index <- sample(1:nrow(train), nrow(train)*0.75)
train_index=1:nrow(train)
# Full data set
data_variables <- as.matrix(train[,-1])
data_label <- train[,1]
data_matrix <- xgb.DMatrix(data = as.matrix(train), 
                           label = data_label)
# split train data and make xgb.DMatrix
train_data   <- data_variables[train_index,]
train_label  <- data_label[train_index]
train_matrix <- xgb.DMatrix(data = train_data, 
                            label = train_label)
# split test data and make xgb.DMatrix
test_data  <- data_variables[-train_index,]
test_label <- data_label[-train_index]
test_matrix <- xgb.DMatrix(data = test_data, label = test_label)


numberOfClasses <- length(unique(train$label))+1
xgb_params <- list("objective" = "multi:softprob",
                   "eval_metric" = "mlogloss",
                   "num_class" = numberOfClasses)
nround    <- 50 # number of XGBoost rounds
cv.nfold  <- 5

# Fit cv.nfold * cv.nround XGB models and save OOF predictions
cv_model <- xgb.cv(params = xgb_params,
                   data = train_matrix, 
                   nrounds = nround,
                   nfold = cv.nfold,
                   verbose = FALSE,
                   prediction = TRUE)
summary(cv_model)

OOF_prediction <- data.frame(cv_model$pred) %>%
  mutate(max_prob = max.col(., ties.method = "last"),
         label = train_label -1)
head(OOF_prediction)

# confusion matrix
confusionMatrix(factor(OOF_prediction$max_prob-2),
                factor(OOF_prediction$label),
                mode = "everything")

bst_model <- xgb.train(params = xgb_params,
                       data = train_matrix,
                       nrounds = 1000)

  # Predict hold-out test set
test_pred <- predict(bst_model, newdata = test_matrix)
test_prediction <- matrix(test_pred, nrow = numberOfClasses,
                          ncol=length(test_pred)/numberOfClasses) %>%
  t() %>%
  data.frame() %>%
  mutate(label = test_label - 1,
         max_prob = max.col(., "last"))
# confusion matrix of test set
confusionMatrix(factor(test_prediction$max_prob-2),
                factor(test_prediction$label),
                mode = "everything")

#----
numberOfClasses <- length(unique(train$label))+1
xgb_params <- list("objective" = "multi:softprob",
                   "eval_metric" = "mlogloss",
                   "num_class" = numberOfClasses)

bst_model <- xgb.train(params = xgb_params,
                       data = train_matrix,
                       nrounds = 100)

test_matrix <- xgb.DMatrix(data = as.matrix(test))
test_pred <- predict(bst_model, newdata = test_matrix)

test_prediction <- matrix(test_pred, nrow = numberOfClasses,
                          ncol=length(test_pred)/numberOfClasses) %>%
  t() %>%
  data.frame() %>%
  mutate(max_prob = max.col(., "last")-2)

A=matrix(0,28001,2)
A[1,1]="ImageId"
A[1,2]="Label"
A[2:28001,1]=1:28000
A[2:28001,2]=test_prediction[,12]
A=data.frame(A)

write.table(A, "XGB14.csv",
            row.names = F, sep = ",",
            col.names = F)
