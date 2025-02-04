library(dplyr)
library(splitstackshape)
library(caret)
library(e1071)
library(rpart)
library(rattle)
library(rpart.plot)
library(partykit)
library(ROCR)
library(tidyverse)
library(dummies)
library(corrplot)
library(pROC)

# Load Data
insurance_data <- read.csv("train.csv", stringsAsFactors = TRUE)

# Data Preprocessing
insurance_data <- insurance_data %>%
  select(-id) %>% 
  mutate(
    Driving_License = as.factor(Driving_License),
    Previously_Insured = as.factor(Previously_Insured),
    Vehicle_Age = as.factor(Vehicle_Age),
    Vehicle_Damage = as.factor(Vehicle_Damage),
    Response = as.factor(Response)
  )

# Normalizing Annual Premium
insurance_data$Annual_Premium <- (insurance_data$Annual_Premium - min(insurance_data$Annual_Premium)) / 
  (max(insurance_data$Annual_Premium) - min(insurance_data$Annual_Premium))

# Check for missing values
colSums(is.na(insurance_data))

# Split Data into Training and Testing
set.seed(100)
index <- sample(nrow(insurance_data), 0.8 * nrow(insurance_data))
data_train <- insurance_data[index, ]
data_test <- insurance_data[-index, ]

# Handle Imbalanced Data with Downsampling
data_train <- downSample(x = data_train %>% select(-Response), 
                         y = data_train$Response, 
                         yname = "Response")

# Logistic Regression Model
logistic_model <- glm(Response ~ ., data = data_train, family = binomial)
summary(logistic_model)

# Logistic Regression Predictions
logistic_probs <- predict(logistic_model, data_test, type = "response")
logistic_preds <- ifelse(logistic_probs > 0.5, 1, 0)

# Logistic Regression Evaluation
logistic_roc <- roc(data_test$Response, logistic_probs)
plot(logistic_roc, col = "blue", main = "ROC Curve - Logistic Regression")
logistic_auc <- auc(logistic_roc)
print(paste("Logistic Regression AUC:", logistic_auc))

# Decision Tree Model
dtree_model <- rpart(Response ~ ., data = data_train, method = "class", cp = 0.01)
fancyRpartPlot(dtree_model)

# Decision Tree Predictions
dtree_preds <- predict(dtree_model, data_test, type = "class")

# Decision Tree Evaluation
dtree_roc <- roc(data_test$Response, as.numeric(dtree_preds))
plot(dtree_roc, col = "red", main = "ROC Curve - Decision Tree")
dtree_auc <- auc(dtree_roc)
print(paste("Decision Tree AUC:", dtree_auc))

# Confusion Matrices
confusionMatrix(as.factor(logistic_preds), data_test$Response, positive = "1")
confusionMatrix(dtree_preds, data_test$Response, positive = "1")
