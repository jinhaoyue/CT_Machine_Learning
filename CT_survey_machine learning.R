####clean data presurvey
if (!require("haven")) install.packages("haven")
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("plyr")) install.packages("plyr")
if (!require("dplyr")) install.packages("dplyr")
if (!require("paran")) install.packages("paran")
if (!require("tidyverse")) install.packages("tidyverse")
if (!require("mmpf")) install.packages("mmpf")
library(haven)
library(ggplot2)
library(plyr)
library(dplyr)
library(paran)
library(tidyverse)
library(mmpf)



dataset <- read_sav("data_merge.sav")
nrow(dataset)


# it should be some percentages low or high
# you want to have a classification
# 
dataset$group <- factor(dataset$group, levels = c("Low", "High"))
str(dataset$group)
summary(dataset$group)

data_classfication <- subset(dataset, select = c(CT_pre_mean, time_train, 
                                                 time_explain, correct_number, 
                                                 number_explain, self_efficacy, 
                                                 difficulty, group))
data_regression <- subset(dataset, select = c(CT_pre_mean, time_train, 
                                                 time_explain, correct_number, 
                                                 number_explain, self_efficacy, 
                                                 difficulty, CT_post_mean))
###########balance the dataset##########
#ovun.sample from ROSE package
## 78 / 0.50 = 156
if (!require("ROSE")) install.packages("ROSE")
library(ROSE)

oversampling_result <- ovun.sample(formula = group ~ ., data = data_classfication, 
				method = "over", N = 156, seed = 123)
head(oversampling_result)
data_balance <- oversampling_result$data 
prop.table(table(data_balance$group))

summary(data_balance)
str(data_balance)
head(data_balance, 10)
data_balance$group <- factor(data_balance$group, levels = c("Low", "High"))
levels(data_balance$group)

if (!require("mlr")) install.packages("mlr")
library(mlr)
glimpse(data_balance)

####split data into training, validation, and test datasets
set.seed(37)
spec = c(train = .7, test = .3)

g = sample(cut(
  seq(nrow(data_balance)), 
  nrow(data_balance)*cumsum(c(0,spec)),
  labels = names(spec)
))

res = split(data_balance, g)

train <- res$train
test <- res$test

#Check dimensions of the split 
prop.table(table(train$group)) * 100
prop.table(table(test$group)) * 100

# Install caretStack
###devtools::install_github("kforthman/caretStack")

library(caretStack)

#3-fold cross validation
#run 500 times
# setting seed to generate a 
# reproducible random sampling
library(tidyverse)
library(caret)
set.seed(124)
 
# define training control which
# generates parameters that further
# control how models are created
train_control <- trainControl(method = "repeatedcv",
                              number = 3, repeats = 500)

##############################################
##############################################
############Naive Bayes classifier############
##############################################
##############################################

set.seed(37)
model_NB <- train(group~., data = train,
                  trControl = train_control, method = "nb")

print(model_NB)
set.seed(37)
model_NB_pred <- predict(model_NB$finalModel, train)
model_NB_pred
(tab_NB <- table(model_NB_pred$class, train$group))
sum(diag(tab_NB)/sum(tab_NB))
set.seed(37)
model_NB_pred_original <- predict(model_NB$finalModel, data_classfication)
tab_NB_original <- table(model_NB_pred_original$class, data_classfication$group)
sum(diag(tab_NB_original)/sum(tab_NB_original))

##ROC and AUC
library(ROCR)
predvec_NB <- model_NB_pred_original$posterior
predvec_NB <- data.frame(predvec_NB)
predvec_NB <- predvec_NB$Low
realvec_NB <- ifelse(data_classfication$group=="High", 0, 1)
head(predvec_NB,5)
pr_NB <- prediction(predvec_NB, realvec_NB)
prf_NB <- ROCR::performance(pr_NB, "tpr", "fpr")
plot(prf_NB, main = "ROC curve - Naive Bayes", sub = "AUC =  0.930", colorize = T)
abline(a = 0, b = 1)

auc_NB <-  ROCR::performance(pr_NB, measure = "auc")
auc_NB <- auc_NB@y.values[[1]]
auc_NB

#criteria for train set
library(caret)
precision(tab_NB)
recall(tab_NB)
sensitivity(tab_NB)
F_meas(tab_NB)
specificity(tab_NB)
#criteria for original set
precision(tab_NB_original)
recall(tab_NB_original)
sensitivity(tab_NB_original)
F_meas(tab_NB_original)
specificity(tab_NB_original)


#criteria for test set
set.seed(37)
model_NB_pred_test <- predict(model_NB$finalModel, test)
tab_NB_test <- table(model_NB_pred_test$class, test$group)
sum(diag(tab_NB_test)/sum(tab_NB_test))

precision(tab_NB_test)
recall(tab_NB_test)
sensitivity(tab_NB_test)
F_meas(tab_NB_test)
specificity(tab_NB_test)

#criteria for balanced data set
set.seed(37)
model_NB_pred_data_balance <- predict(model_NB$finalModel, data_balance)
tab_NB_data_balance <- table(model_NB_pred_data_balance$class, data_balance$group)
sum(diag(tab_NB_data_balance)/sum(tab_NB_data_balance))

precision(tab_NB_data_balance)
recall(tab_NB_data_balance)
sensitivity(tab_NB_data_balance)
F_meas(tab_NB_data_balance)
specificity(tab_NB_data_balance)

##############################################
##############################################
#############Logistic Regression #############
##############################################
##############################################
set.seed(37)
model_LR <- train(group~., data = train, 
                  trControl = train_control, method = 'glmnet', family = "binomial")

model_LR_pred_original_class <- predict(model_LR, data_classfication)
model_LR_pred_class <- predict(model_LR, train)
model_LR_pred_prob <- predict(model_LR, train, "prob")
model_LR_pred_original_prob <- predict(model_LR, data_classfication, "prob")


(tab_LR <- table(model_LR_pred_class, train$group))
sum(diag(tab_LR)/sum(tab_LR))
tab_LR_original <- table(model_LR_pred_original_class, data_classfication$group)
sum(diag(tab_LR_original)/sum(tab_LR_original))
##ROC and AUC
predvec_LR <-  model_LR_pred_original_prob
predvec_LR <- data.frame(predvec_LR)
predvec_LR <- predvec_LR$Low
realvec_LR <- ifelse(data_classfication$group=="High", 0, 1)
head(predvec_LR,5)
pr_LR <- prediction(predvec_LR, realvec_LR)
prf_LR <- ROCR::performance(pr_LR, "tpr", "fpr")
plot(prf_LR, main = "ROC curve - Logistic regression", sub = "AUC =  0.900", colorize = T)
abline(a = 0, b = 1)

auc_LR <- ROCR::performance(pr_LR, measure = "auc")
auc_LR <- auc_LR@y.values[[1]]
auc_LR

#criteria for train set
precision(tab_LR)
recall(tab_LR)
sensitivity(tab_LR)
F_meas(tab_LR)
specificity(tab_LR)

#criteria for original set
precision(tab_LR_original)
recall(tab_LR_original)
sensitivity(tab_LR_original)
F_meas(tab_LR_original)
specificity(tab_LR_original)


#criteria for test set
set.seed(37)
model_LR_pred_class_test <- predict(model_LR, test)
(tab_LR_test <- table(model_LR_pred_class_test, test$group))
sum(diag(tab_LR_test)/sum(tab_LR_test))

precision(tab_LR_test)
recall(tab_LR_test)
sensitivity(tab_LR_test)
F_meas(tab_LR_test)
specificity(tab_LR_test)

#criteria for balanced data set
model_LR_pred_class_data_balance <- predict(model_LR, data_balance)
(tab_LR_data_balance <- table(model_LR_pred_class_data_balance, data_balance$group))
sum(diag(tab_LR_data_balance)/sum(tab_LR_data_balance))

precision(tab_LR_data_balance)
recall(tab_LR_data_balance)
sensitivity(tab_LR_data_balance)
F_meas(tab_LR_data_balance)
specificity(tab_LR_data_balance)

##############################################
#########developing models: DT ###############
##############################################
set.seed(37)
model_DT <- train(group~., data = train, 
                  trControl = train_control, method = 'rpart', tuneLength = 20)
model_DT
summary(model_DT$finalModel)
varImp(model_DT)$importance
# plot the model
plot(model_DT$finalModel, uniform=TRUE,
     main="Classification Tree")
text(model_DT$finalModel, use.n.=TRUE, all=TRUE, cex=.8)


suppressMessages(library(rattle))
fancyRpartPlot(model_DT$finalModel)

## predict
set.seed(37)
model_DT_pred <- predict(model_DT$finalModel, train)
model_DT_pred <- data.frame(model_DT_pred)
model_DT_pred_class <- ifelse(model_DT_pred$Low > 0.5,"Low", "High" )
model_DT_pred_class <- factor(model_DT_pred_class, levels = c("Low", "High"))
(tab_DT <- table(model_DT_pred_class, train$group))
sum(diag(tab_DT)/sum(tab_DT))

set.seed(37)
model_DT_pred_original <- predict(model_DT$finalModel, data_classfication)
model_DT_pred_original <- data.frame(model_DT_pred_original)
model_DT_pred_original_class <- ifelse(model_DT_pred_original$Low > 0.5,"Low", "High" )
model_DT_pred_original_class <- factor(model_DT_pred_original_class, levels = c("Low", "High"))
tab_DT_original <- table(model_DT_pred_original_class, data_classfication$group)
sum(diag(tab_DT)/sum(tab_DT))
sum(diag(tab_DT_original)/sum(tab_DT_original))


##ROC and AUC
predvec_DT <- model_DT_pred_original$Low
head(predvec_DT, 5)

realvec_DT <- ifelse(data_classfication$group=="High", 0, 1)
pr_DT <- prediction(predvec_DT, realvec_DT)
prf_DT <- ROCR::performance(pr_DT, "tpr", "fpr")
plot(prf_DT, main = "ROC curve - DT", sub = "AUC =  0.950", colorize = T)
abline(a = 0, b = 1)

auc_DT <- ROCR::performance(pr_DT, measure = "auc")
auc_DT <- auc_DT@y.values[[1]]
auc_DT

#criteria for train set
precision(tab_DT)
recall(tab_DT)
sensitivity(tab_DT)
F_meas(tab_DT)
specificity(tab_DT)


#criteria for original set
precision(tab_DT_original)
recall(tab_DT_original)
sensitivity(tab_DT_original)
F_meas(tab_DT_original)
specificity(tab_DT_original)


#criteria for test set

model_DT_pred_test <- predict(model_DT$finalModel, test)
model_DT_pred_test <- data.frame(model_DT_pred_test)
model_DT_pred_test_class <- ifelse(model_DT_pred_test$Low > 0.5,"Low", "High" )
model_DT_pred_test_class <- factor(model_DT_pred_test_class, levels = c("Low", "High"))
tab_DT_test <- table(model_DT_pred_test_class, test$group)
sum(diag(tab_DT_test)/sum(tab_DT_test))

precision(tab_DT_test)
recall(tab_DT_test)
sensitivity(tab_DT_test)
F_meas(tab_DT_test)
specificity(tab_DT_test)

#criteria for balanced data set

model_DT_pred_data_balance <- predict(model_DT$finalModel, data_balance)
model_DT_pred_data_balance <- data.frame(model_DT_pred_data_balance)
model_DT_pred_data_balance_class <- ifelse(model_DT_pred_data_balance$Low > 0.5,"Low", "High" )
model_DT_pred_data_balance_class <- factor(model_DT_pred_data_balance_class, levels = c("Low", "High"))

tab_DT_data_balance <- table(model_DT_pred_data_balance_class, data_balance$group)
sum(diag(tab_DT_data_balance)/sum(tab_DT_data_balance))

precision(tab_DT_data_balance)
recall(tab_DT_data_balance)
sensitivity(tab_DT_data_balance)
F_meas(tab_DT_data_balance)
specificity(tab_DT_data_balance)

##############################################
#########developing models: knn ##############
##############################################
set.seed(37)
model_KNN <- train(group~., data = train, 
                  trControl = train_control, method = 'knn', preProcess = c("center","scale"), tuneLength = 20)
model_KNN


## predict
set.seed(37)
model_KNN_pred <- predict(model_KNN, train)
model_KNN_pred

(tab_KNN <- table(model_KNN_pred, train$group))
sum(diag(tab_KNN)/sum(tab_KNN))

set.seed(37)
model_KNN_origninal_pred <- predict(model_KNN, data_classfication)
(tab_original_KNN <- table(model_KNN_origninal_pred, data_classfication$group))
sum(diag(tab_original_KNN)/sum(tab_original_KNN))

model_KNN_pred_prob <- predict(model_KNN, data_classfication, type = "prob")
model_KNN_pred_prob <- data.frame(model_KNN_pred_prob)


##ROC and AUC
predvec_KNN <- model_KNN_pred_prob$Low
head(predvec_KNN, 5)

realvec_KNN <- ifelse(data_classfication$group=="High", 0, 1)
pr_KNN <- prediction(predvec_KNN, realvec_KNN)
prf_KNN <- ROCR::performance(pr_KNN, "tpr", "fpr")
plot(prf_KNN, main = "ROC curve - DT", sub = "AUC =  0.862", colorize = T)
abline(a = 0, b = 1)

auc_KNN <- ROCR::performance(pr_KNN, measure = "auc")
auc_KNN <- auc_KNN@y.values[[1]]
auc_KNN

#criteria for train set
precision(tab_KNN)
recall(tab_KNN)
sensitivity(tab_KNN)
F_meas(tab_KNN)
specificity(tab_KNN)

#criteria for original set
precision(tab_original_KNN)
recall(tab_original_KNN)
sensitivity(tab_original_KNN)
F_meas(tab_original_KNN)
specificity(tab_original_KNN)

#criteria for test set
model_KNN_test_pred <- predict(model_KNN, test)
(tab_test_KNN <- table(model_KNN_test_pred, test$group))
sum(diag(tab_test_KNN)/sum(tab_test_KNN))


precision(tab_test_KNN)
recall(tab_test_KNN)
sensitivity(tab_test_KNN)
F_meas(tab_test_KNN)
specificity(tab_test_KNN)

#criteria for balance set
model_KNN_data_balance_pred <- predict(model_KNN, data_balance)
(tab_data_balance_KNN <- table(model_KNN_data_balance_pred, data_balance$group))
sum(diag(tab_data_balance_KNN)/sum(tab_data_balance_KNN))


precision(tab_data_balance_KNN)
recall(tab_data_balance_KNN)
sensitivity(tab_data_balance_KNN)
F_meas(tab_data_balance_KNN)
specificity(tab_data_balance_KNN)


################################
################################
############ALL ROC#############
################################
################################
preds_list <- list(predvec_LR, predvec_NB, predvec_DT, predvec_KNN)

# List of actual values (same for all)
m <- length(preds_list)
actuals_list <- rep(list(realvec_LR), m)

# Plot the ROC curves
pred <- prediction(preds_list, actuals_list)
rocs <- ROCR::performance(pred, "tpr", "fpr")
plot(rocs, col = as.list(1:m), main = "Test Set ROC Curves") + abline(coef = c(0,1))
legend(x = "bottomright", 
       legend = c("LR: AUC = 0.900", "NB: AUC = 0.930","DT: AUC = 0.950", "KNN: AUC = 0.862"),
       fill = 1:m)
auc_LR
auc_NB
auc_DT
auc_KNN