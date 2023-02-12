if (!require("ROCR")) install.packages("ROCR")
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("caret")) install.packages("caret")
if (!require("rpart")) install.packages("rpart")

library(ROCR)
library(ggplot2)
library(lattice)
library(caret)
library(rpart)

data <- read.csv("data_clean.csv", header=TRUE)
data
nrow(data)
data$group <- factor(data$group, levels = c("Low", "High"))
str(data$group)

# Cross-validation Approach

#####################################
# DECISION TREE
#####################################

###########
# LOOCV
###########
# Set seed
set.seed(42)
# Accuracy = 0.8387097
library(caret)
model_DT <- train(
  group ~ ., data,
  method = "rpart",
  trControl = trainControl(
    method = "LOOCV"
  )
)
model_DT
# Out-of-sample error: Accuracy: 0.8172; Balanced Accuracy: 0.5949
cm_DT <- confusionMatrix(model_DT$pred$pred, model_DT$pred$obs)
cm_DT
F_meas(cm_DT$table) # 0.32
(tab_DT <- table(model_DT$pred$pred, model_DT$pred$obs))
(accuracy <- sum(diag(tab_DT)/sum(tab_DT)))

# Variable importance
varImp(model_DT, surrogates = TRUE, competes = FALSE)
#varImp(model_DT$finalModel, surrogates = TRUE, competes = FALSE) #same as for model_DT

# Overall
# time_train     100.000
# difficulty      93.535
# time_explain    20.543
# CT_pre_mean     10.342
# correct_number   9.253
# number_explain   8.709
# self_efficacy    0.000

# Plot
suppressMessages(library(rattle))
library(rpart.plot)
rpart.plot(model_DT$finalModel)
fancyRpartPlot(model_DT$finalModel)

#####################################
# LOGISTIC REGRESSION 
#####################################

###########
# LOOCV
###########

# Build logistic regression model

# Set seed
set.seed(42)
# Accuracy = 0.827957
library(caret)
model_LR <- train(
  group ~ ., data,
  method = "glm",
  family = "binomial",
  trControl = trainControl(
    method = "LOOCV"
  )
)
model_LR

# Out-of-sample error: Accuracy: 0.828; Balanced Accuracy: 0.6282
cm_LR <- confusionMatrix(model_LR$pred$pred, model_LR$pred$obs)
cm_LR
F_meas(cm_LR$table) # 0.3846154
(tab_LR <- table(model_LR$pred$pred, model_LR$pred$obs))
(accuracy <- sum(diag(tab_LR)/sum(tab_LR)))

############################
# Naive Bayes
############################

###########
# LOOCV
###########

library(klaR)

# Set seed
set.seed(42)
# Accuracy = 0.8172043
library(caret)
model_NB <- train(
  group ~ ., data,
  method = "nb",
  trControl = trainControl(
    method = "LOOCV"
  )
)
model_NB

# Out-of-sample error: Accuracy: 0.8118; Balanced Accuracy: 0.7128
cm_NB <- confusionMatrix(model_NB$pred$pred, model_NB$pred$obs)
cm_NB
F_meas(cm_NB$table) # 0.4927536
(tab_NB <- table(model_NB$pred$pred, model_NB$pred$obs))
(accuracy <- sum(diag(tab_NB)/sum(tab_NB)))

###################################
# KNN
###################################

#########
# LOOCV
#########

# Set seed
set.seed(42)
# Accuracy = 0.8709677; k = 7
library(caret)
model_KNN <- train(
  group ~ ., data,
  method = "knn",
  #preProcess = c("center","scale"), tuneLength = 20,
  trControl = trainControl(
    method = "LOOCV"
  )
)
model_KNN

# Out-of-sample error: Accuracy: 0.853; Balanced Accuracy: 0.6342
cm_KNN <- confusionMatrix(model_KNN$pred$pred, model_KNN$pred$obs)
cm_KNN
F_meas(cm_KNN$table) # 0.4057971

#############################
# Validation-set approach
#############################
# Set seed
set.seed(1)

# Split data in train and test set
# Create a partition index with 80% of the data to create a training set and stratify the partitions by the response variable group.
# Aims to balance the class distributions within the splits. Tries to ensure a split that has a similar distribution of the supplied variable in both datasets.
library(caret)
index <- createDataPartition(data$group, p = 0.8, list = FALSE)

# Build logistic regression model

train <- data[index,]
nrow(train)
test <- data[-index,]
nrow(test)

table(train$group)
table(test$group)

# SMOTE
if (!require("ROSE")) install.packages("ROSE")
library(ROSE)
oversampling_result <- ovun.sample(formula = group ~ ., data = train,
                                   method = "over", N = 126, seed = 123)
head(oversampling_result)
data_balance <- oversampling_result$data 
prop.table(table(data_balance$group))

table(data_balance$group)

#if (!require("mlr")) install.packages("mlr")
#library(mlr)
#glimpse(data_balance)

###############################
# Naive Bayes - Train on balanced subset (train, which is 80% of the original file), test on unbalanced remaining subset (test, which is 20% of the original file)
###############################

# Set seed
set.seed(42)
# Accuracy = 0.8492063
if (!require("caret")) install.packages("caret")
library(caret)
model_NB <- train(
  group ~ ., data_balance,
  method = "nb",
  trControl = trainControl(
    method = "LOOCV"
  )
)
model_NB

# Out-of-sample error: Accuracy: 0.7222; Balanced Accuracy: 0.7000
model_NB_pred <- predict(model_NB$finalModel, test)
model_NB_pred_prob <- predict(model_NB$finalModel, test, type="prob")
model_NB_pred
(tab_NB <- table(model_NB_pred$class, test$group))
(accuracy <- sum(diag(tab_NB)/sum(tab_NB))) # Accuracy
cm_NB <- confusionMatrix(model_NB_pred$class, test$group)
cm_NB
F_meas(cm_NB$table) # 0.4444444
precision(cm_NB$table) #0.3333333
##ROC and AUC
library(ROCR)
predvec_NB <- data.frame(model_NB_pred_prob)$posterior.Low
realvec_NB <- ifelse(test$group=="High", 0, 1)
pr_NB <- prediction(predvec_NB, realvec_NB)
prf_NB <- ROCR::performance(pr_NB, "tpr", "fpr")
plot(prf_NB, main = "ROC curve - Naive Bayes", sub = "AUC", colorize = T)
abline(a = 0, b = 1)
 
auc_NB <-  ROCR::performance(pr_NB, measure = "auc")
auc_NB <- auc_NB@y.values[[1]]
auc_NB # 0.8222222

# In-sample error: (Balanced) Accuracy: 0.8968
model_NB_train_pred <- predict(model_NB$finalModel, data_balance)
model_NB_train_pred
(tab_train_NB <- table(model_NB_train_pred$class, data_balance$group))
(accuracy <- sum(diag(tab_train_NB)/sum(tab_train_NB))) # Accuracy
cm_train_NB <- confusionMatrix(model_NB_pred$class, data_balance$group)
cm_train_NB
F_meas(cm_train_NB$table) # 0.8849558
precision(cm_train_NB$table) #1
###############################################
# Logistic Regression
###############################################
#if (!require("caretStack")) install.packages("caretStack")
#library(caretStack)
#if (!require("tidyverse")) install.packages("tidyverse")
#library(tidyverse)
#library(caret)

# Set seed
set.seed(42)
# Accuracy: 0.8196721
model_LR <- train(
  group ~ ., data_balance,
  method = "glm",
  family = "binomial",
  trControl = trainControl(
    method = "LOOCV"
  )
)
summary(model_LR)

# Out-of-sample error: Accuracy: 0.7778; Balanced Accuracy: 0.7333
glm.pred <- predict(model_LR, test) # Predictions
(tab_LR <- table(test$group, glm.pred))
(accuracy <- sum(diag(tab_LR)/sum(tab_LR)))
cm_LR <- confusionMatrix(glm.pred, test$group)
cm_LR
F_meas(cm_LR$table) # 0.5
precision(cm_LR$table) #0.4
# Accuracy:
missing_classerr <- mean(glm.pred != test$group)
print(paste('Accuracy =', 1 - missing_classerr))

#library(pROC)
#test_prob = predict(model_LR, newdata = test, type = "prob")
#test_roc = roc(test$group ~ test_prob, plot = TRUE, print.auc = TRUE)
#as.numeric(test_roc$auc)

# One way to draw ROC/AUC
library(ROCR)
glm.prob <-  predict(model_LR, test, type = "prob") # Predictions prob
pred <- prediction(glm.prob$Low, test$group)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
unlist(slot(performance(pred, "auc"), "y.values")) # AUC = 0.91

## Another way: ROC and AUC
model_LR_pred_prob <- predict(model_LR, test, type = "prob")
predvec_LR <-  model_LR_pred_prob
predvec_LR <- data.frame(predvec_LR)
predvec_LR <- predvec_LR$Low
realvec_LR <- ifelse(test$group=="High", 0, 1)
pr_LR <- prediction(predvec_LR, realvec_LR)
prf_LR <- ROCR::performance(pr_LR, "tpr", "fpr")
plot(prf_LR, main = "ROC curve - Logistic regression", sub = "AUC", colorize = T)
abline(a = 0, b = 1)

auc_LR <- ROCR::performance(pr_LR, measure = "auc")
auc_LR <- auc_LR@y.values[[1]]
auc_LR # 0.911

# In-sample error: (Balanced) Accuracy: 0.8492
glm.train_pred <- predict(model_LR, data_balance) # Predictions
(tab_train_LR <- table(data_balance$group, glm.train_pred))
(accuracy <- sum(diag(tab_train_LR )/sum(tab_train_LR )))
cm_train_LR <- confusionMatrix(glm.train_pred, data_balance$group)
cm_train_LR
F_meas(cm_train_LR$table) # 0.8429752
precision(cm_train_LR$table) #0.8793103
#####################################
# DT
#####################################

# Set seed
set.seed(42)
# Accuracy = 0.8968254
library(caret)
model_DT <- train(
  group ~ ., data_balance,
  method = "rpart", tuneLength = 30,
  trControl = trainControl(
    method = "LOOCV"
  )
)
model_DT

# Variable importance
varImp(model_DT, surrogates = TRUE, competes = FALSE)
#varImp(model_DT$finalModel, surrogates = TRUE, competes = FALSE) # same as for model_DT

# Overall
# time_train     100.000
# CT_pre_mean     61.562
# difficulty      48.389

suppressMessages(library(rattle))
fancyRpartPlot(model_DT$finalModel)

# Out-of-sample error: Accuracy: 0.8889; Balanced Accuracy: 0.9333
model_DT_pred = predict(model_DT, test)
model_DT_pred
(tab_DT <- table(model_DT_pred, test$group))
(accuracy <- sum(diag(tab_DT)/sum(tab_DT))) # Accuracy
cm_DT <- confusionMatrix(model_DT_pred, test$group)
cm_DT
F_meas(cm_DT$table) # 0.75
precision(cm_DT$table) #0.6
(error.rate = round(mean(model_DT_pred != test$group),2)) # (= 1 - Accuracy)

# ## ROC and AUC
model_DT_pred_prob <- predict(model_DT, test, type = "prob")
predvec_DT <- data.frame(model_DT_pred_prob)$Low
realvec_DT <- ifelse(test$group=="High", 0, 1)
pr_DT <- prediction(predvec_DT, realvec_DT)
prf_DT <- ROCR::performance(pr_DT, "tpr", "fpr")
plot(prf_DT, main = "ROC Curve - Decision Tree", sub = "AUC", colorize = T)
abline(a = 0, b = 1)

auc_DT <- ROCR::performance(pr_DT, measure = "auc")
auc_DT <- auc_DT@y.values[[1]]
auc_DT # 0.9666667

# In-sample error: (Balanced) Accuracy: 0.8968
model_DT_train_pred = predict(model_DT, data_balance)
model_DT_train_pred
(tab_train_DT <- table(model_DT_train_pred, data_balance$group))
(accuracy <- sum(diag(tab_train_DT)/sum(tab_train_DT))) # Accuracy
cm_train_DT <- confusionMatrix(model_DT_train_pred, data_balance$group)
cm_train_DT
F_meas(cm_train_DT$table) # 0.8943089
precision(cm_train_DT$table) #0.9166667
(error.rate = round(mean(model_DT_train_pred != data_balance$group),2)) # (= 1 - Accuracy)

###################################
# KNN
###################################

# Set seed
set.seed(42)
# Accuracy = 0.7704918; k = 5
library(caret)
model_KNN <- train(
  group ~ ., data_balance,
  method = "knn", tuneLength = 30,
  trControl = trainControl(
    method = "LOOCV"
  )
)
model_KNN

# Out-of-sample error: Accuracy: 0.8889; Balanced Accuracy: 0.9333
model_KNN_pred = predict(model_KNN, test)
model_KNN_pred
(tab_KNN <- table(model_KNN_pred, test$group))
(accuracy <- sum(diag(tab_KNN)/sum(tab_KNN)))
cm_KNN <- confusionMatrix(model_KNN_pred, test$group)
cm_KNN
F_meas(cm_KNN$table) # 0.75
precision(cm_KNN$table) #0.6
## ROC and AUC
model_KNN_pred_prob <- predict(model_KNN, test, type = "prob")
predvec_KNN <- data.frame(model_KNN_pred_prob)$Low
realvec_KNN <- ifelse(test$group=="High", 0, 1)
pr_KNN <- prediction(predvec_KNN, realvec_KNN)
prf_KNN <- ROCR::performance(pr_KNN, "tpr", "fpr")
plot(prf_KNN, main = "ROC Curve - KNN", sub = "AUC", colorize = T)
abline(a = 0, b = 1)

auc_KNN <- ROCR::performance(pr_KNN, measure = "auc")
auc_KNN <- auc_KNN@y.values[[1]]
auc_KNN # 0.9

# In-sample error: (Balanced) Accuracy: 0.8175
model_KNN_train_pred = predict(model_KNN, data_balance)
model_KNN_train_pred
(tab_train_KNN <- table(model_KNN_train_pred, data_balance$group))
(accuracy <- sum(diag(tab_train_KNN)/sum(tab_train_KNN)))
cm_train_KNN <- confusionMatrix(model_KNN_train_pred, data_balance$group)
cm_train_KNN
F_meas(cm_train_KNN$table) # 0.7927928
precision(cm_train_KNN$table) #0.9166667
################################
############ ALL ROC ###########
################################
preds_list <- list(predvec_LR, predvec_NB, predvec_DT, predvec_KNN)
# 
# List of actual values (same for all)
m <- length(preds_list)
actuals_list <- rep(list(realvec_LR), m)

# Plot the ROC curves
pred <- prediction(preds_list, actuals_list)
rocs <- ROCR::performance(pred, "tpr", "fpr")
plot(rocs, col = as.list(1:m), main = "Test Set ROC Curves") + abline(coef = c(0,1))
legend(x = "bottomright", 
        legend = c("LR: AUC = 0.91", "NB: AUC = 0.82","DT: AUC = 0.97", "KNN: AUC = 0.90"),
        fill = 1:m)
auc_LR
auc_NB
auc_DT
auc_KNN
