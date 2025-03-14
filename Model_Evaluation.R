library("dplyr")
library("VGAM")
library("nnet")
library(caret)
library(class)
library(e1071)
set.seed(42)
df <- read.csv("dataset.csv", sep=";", header=TRUE)
df <- df[sample(nrow(df)), ]
X <- df[, !(names(df) %in% c("ShotType"))]
y <- df["ShotType"]

df %>% group_by(df$ShotType) %>% summarize(count=n())
#above head appears 3055 times and it is most frequent
#tip-in appears 61 times and it is the least frequent
#Try stratified CV to keep the distributions of shot types

#LOG-LOSS
log_loss <- function(probabilities, y_true){
  N <-  length(y_true)
  loss <- 0
  for (i in 1:N){
    loss <- loss - (1/N) * log(probabilities[i, y_true[i]] + 1e-14)
  }
  return(loss)
}

#ACCURACY
accuracy <- function(predictions, y_true){
  return(mean(predictions == y_true))
}


train_baseline_classifier <- function(y_train){
  return(prop.table(table(y_train)))
}

#USE TO PREDICT LABELS - ACCURACY
probabilities <- train_baseline_classifier(y)
predict_baseline_classifier <- function(probabilities, n=1){
  return(sample(names(probabilities), size = n, replace = TRUE, prob = probabilities))
}

#USE TO RETURN A MATRIX OF PROBABILITIES - LOG-LOSS
baseline_classifier_probabilities <- function(probabilities, n=1){
  probability_matrix <- matrix(rep(probabilities, n), nrow = n, byrow = TRUE)
  colnames(probability_matrix) <- names(probabilities)
  return(probability_matrix)
}

train_i <- as.integer(nrow(df)*0.8)
test_i <- train_i + 1
#LOGISTIC REGRESSION#
df$ShotType <- as.factor(df$ShotType)
train_data <- df[1:train_i,]
test_data <- df[test_i:nrow(df),]
levels(train_data$ShotType)
levels(test_data$ShotType)
model <- multinom(ShotType ~ ., family = multinomial, data = train_data)

prediction_probabilities <- predict(model, newdata = test_data, type = "probs")
prediction_labels <- predict(model, newdata = test_data, type = "class")

log_loss(prediction_probabilities, test_data$ShotType)
accuracy(prediction_labels, test_data$ShotType)

baseline_probs <- train_baseline_classifier(train_data$ShotType)
baseline_prediction_probabilities <- baseline_classifier_probabilities(baseline_probs, nrow(test_data))
baseline_prediction_labels <- predict_baseline_classifier(baseline_probs, nrow(test_data))

log_loss(baseline_prediction_probabilities, test_data$ShotType)
accuracy(baseline_prediction_labels, test_data$ShotType)

###TESTING KNN
y_train <- train_data$ShotType
y_test  <- test_data$ShotType

train_features <- train_data[ , !(names(train_data) == "ShotType")]
test_features  <- test_data[ , !(names(test_data) == "ShotType")]


dummies_model <- dummyVars(~ ., data = train_features, fullRank = TRUE)

# Apply to both sets
X_train <- predict(dummies_model, newdata = train_features)
X_test  <- predict(dummies_model, newdata = test_features, na.action = na.pass)

scaler <- preProcess(X_train, method = c("center", "scale"))
X_train_scaled <- predict(scaler, X_train)
X_test_scaled  <- predict(scaler, X_test)

knn_model <- knn3(x = X_train_scaled, y = y_train, k = 100)
probabilities_knn <- predict(knn_model, X_test, type="prob")
labels_knn <- predict(knn_model, X_test_scaled, type="class")

log_loss(probabilities_knn, y_test)
accuracy(labels_knn, y_test)

head(probabilities_knn)

svm_model <- svm(x = X_train_scaled, y = y_train, 
                 probability = TRUE, kernel = "radial", gamma=0.5)  # or "linear", etc.

# 4. Predict probabilities
svm_pred <- predict(svm_model, newdata = X_test_scaled, probability = TRUE)
svm_probs <- attr(svm_pred, "probabilities")

# 5. Evaluate
log_loss(svm_probs, y_test)
accuracy(svm_pred, y_test)

unique(df$ShotType)

