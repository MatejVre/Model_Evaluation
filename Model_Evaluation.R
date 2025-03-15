library("dplyr")
library("VGAM")
library("nnet")
library(caret)
library(class)
library(e1071)
library(glue)
set.seed(42)
df <- read.csv("dataset.csv", sep=";", header=TRUE)
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
    probability <- probabilities[i, as.character(y_true[i])]
    probability <- max(probability, 1e-15)
    loss <- loss - (1/N) * log(probability)
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

stratified_folds <- function(data, target_column, k=5){
  y <- data[[target_column]]
  labels <-  unique(y)
  folds <-  vector("list", length = k)
  
  for (lab in labels){
    label_fold <- split(sample(which(y == lab)), 
                        rep(1:k, length.out = length(which(y == lab))))
    for (i in 1:k){
      folds[[i]] <- c(folds[[i]], label_fold[[i]])
    }
  }
  return(folds)
}

test_train_split <- function(data, train_size = 0.8){
  train_indices <- sample(1:nrow(data), size = floor(train_size * nrow(data)))
  
  train_set <- data[train_indices, ]
  test_set <- data[-train_indices,]
  return(list(train = train_set, test = test_set))
}

num_folds <- 5
fold_indices <- stratified_folds(df, "ShotType", k=num_folds)

set.seed(42)
for (i in 1:num_folds){
  df$ShotType <- as.factor(df$ShotType)
  test_indices <- fold_indices[[i]]
  
  train_data <- df[-test_indices,]
  test_data <- df[test_indices,]
  
  baseline_probs <- train_baseline_classifier(train_data$ShotType)
  baseline_prediction_probabilities <- baseline_classifier_probabilities(baseline_probs, nrow(test_data))
  baseline_prediction_labels <- predict_baseline_classifier(baseline_probs, nrow(test_data))
  
  print(glue("Log loss: {log_loss(baseline_prediction_probabilities, test_data$ShotType)}"))
  print(glue("Accuracy: {accuracy(baseline_prediction_labels, test_data$ShotType)}"))
  
}
for (i in 1:num_folds){
  df$ShotType <- as.factor(df$ShotType)
  
  test_indices <- fold_indices[[i]]
  
  train_data <- df[-test_indices,]
  test_data <- df[test_indices,]
  
  model <- multinom(ShotType ~ ., family = multinomial, data = train_data, trace = FALSE)
  
  prediction_probabilities <- predict(model, newdata = test_data, type = "probs")
  prediction_labels <- predict(model, newdata = test_data, type = "class")
  
  print(glue("Log loss:{log_loss(prediction_probabilities, test_data$ShotType)}"))
  print(glue("Accuracy: {accuracy(prediction_labels, test_data$ShotType)}"))
}


gamma_values <- c(0.0001, 0.001, 0.01, 0.1, 1)
#This function finds the best gamma value over all folds BUT
#evaluates on the test set.
SVM_CV_flat <- function(df, fold_indices, gamma_values, num_folds){
  best_gamma <- NA
  best_loss <- Inf
  acc_for_best_loss <- 0
  for (gam in gamma_values){
    loss <- 0
    acc <- 0
    print(glue("Testing gamma: {gam}"))
    for (i in 1:num_folds){
      
      df$ShotType <- as.factor(df$ShotType)
      test_indices <- fold_indices[[i]]
      
      train_data <- df[-test_indices,]
      test_data <- df[test_indices,]
      
      y_train <- train_data$ShotType
      y_test  <- test_data$ShotType
      
      train_features <- train_data[ , !(names(train_data) == "ShotType")]
      test_features  <- test_data[ , !(names(test_data) == "ShotType")]
      
      
      dummies_model <- dummyVars(~ ., data = train_features, fullRank = TRUE)
      
      X_train <- predict(dummies_model, newdata = train_features)
      X_test  <- predict(dummies_model, newdata = test_features, na.action = na.pass)
      
      scaler <- preProcess(X_train, method = c("center", "scale"))
      X_train_scaled <- predict(scaler, X_train)
      X_test_scaled  <- predict(scaler, X_test)
      
      svm_model <- svm(x = X_train_scaled, y = y_train, 
                       probability = TRUE, kernel = "radial", gamma=gam)
      
      svm_pred <- predict(svm_model, newdata = X_test_scaled, probability = TRUE)
      svm_probs <- attr(svm_pred, "probabilities")
      
      loss <- loss + log_loss(svm_probs, y_test)
      acc <- acc + accuracy(svm_pred, y_test)
    }
    if (loss < best_loss){
      best_loss <- loss
      best_gamma <- gam
      acc_for_best_loss <- acc
    }
  }
return(list(loss = best_loss / num_folds, accuracy = acc_for_best_loss / num_folds,
            gamma = best_gamma))
}

#This function finds the best parameter for each fold evaluated on the train set
#I think this is the way to go for part 1, but who knows at this point.
SVM_CV_per_fold_tuning <- function(df, fold_indices, gamma_values, num_folds) {
  total_loss <- 0
  total_acc <- 0
  
  df$ShotType <- as.factor(df$ShotType)
  
  for (i in 1:num_folds) {
    
    test_indices <- fold_indices[[i]]
    train_data <- df[-test_indices, ]
    test_data  <- df[test_indices, ]

    y_train <- train_data$ShotType
    y_test  <- test_data$ShotType
    
    train_features <- train_data[, !(names(train_data) == "ShotType")]
    test_features  <- test_data[, !(names(test_data) == "ShotType")]
    
    dummies_model <- dummyVars(~ ., data = train_features, fullRank = TRUE)
    X_train <- predict(dummies_model, newdata = train_features)
    X_test  <- predict(dummies_model, newdata = test_features, na.action = na.pass)
    
    scaler <- preProcess(X_train, method = c("center", "scale"))
    X_train_scaled <- predict(scaler, X_train)
    X_test_scaled  <- predict(scaler, X_test)
    
    best_gamma <- NA
    best_fold_loss <- Inf
    
    for (gam in gamma_values) {
      svm_model <- svm(x = X_train_scaled, y = y_train,
                              probability = TRUE, kernel = "radial", gamma = gam)
      
      pred <- predict(svm_model, newdata = X_train_scaled, probability = TRUE)
      probs <- attr(pred, "probabilities")

      fold_loss <- log_loss(probs, y_train)
      
      if (fold_loss < best_fold_loss) {
        best_fold_loss <- fold_loss
        best_gamma <- gam
      }
    }
    
    final_model <- svm(x = X_train_scaled, y = y_train,
                              probability = TRUE, kernel = "radial", gamma = best_gamma)
    
    final_preds <- predict(final_model, newdata = X_test_scaled, probability = TRUE)
    final_probs <- attr(final_preds, "probabilities")
    
    fold_loss <- log_loss(final_probs, y_test)
    fold_acc <- accuracy(final_preds, y_test)
    
    total_loss <- total_loss + fold_loss
    total_acc <- total_acc + fold_acc
  }
  
  return(list(
    mean_log_loss = total_loss / num_folds,
    mean_accuracy = total_acc / num_folds
  ))
}

SVM_CV_nested <- function(df, fold_indices, gamma_values, num_folds) {
  final_loss <- 0
  best_gamma <- NA
  for (i in 1:num_folds){
    
    best_loss <- Inf
    test_indices <- fold_indices[[i]]
    train_data <- df[-test_indices, ]
    test_data  <- df[test_indices, ]
    
    for (gam in gamma_values){
      
      loss <- 0
      
      inner_fold_indices <- stratified_folds(train_data, "ShotType", 3)
      for (u in 1:length((inner_fold_indices))){
        
        inner_test_indices <- inner_fold_indices[[u]]
        inner_train_data <- train_data[-inner_test_indices, ]
        inner_test_data  <- train_data[inner_test_indices, ]
        
        y_train_inner <- inner_train_data$ShotType
        y_test_inner  <- inner_test_data$ShotType
        
        inner_train_features <- inner_train_data[, !(names(train_data) == "ShotType")]
        inner_test_features  <- inner_test_data[, !(names(test_data) == "ShotType")]
        
        dummies_model <- dummyVars(~ ., data = inner_train_features, fullRank = TRUE)
        X_train_inner <- predict(dummies_model, newdata = inner_train_features)
        X_test_inner  <- predict(dummies_model, newdata = inner_test_features, na.action = na.pass)
        
        scaler <- preProcess(X_train_inner, method = c("center", "scale"))
        X_train_scaled_inner <- predict(scaler, X_train_inner)
        X_test_scaled_inner  <- predict(scaler, X_test_inner)
        
        svm_model <- svm(x = X_train_scaled_inner, y = y_train_inner,
                         probability = TRUE, kernel = "radial", gamma = gam)
        
        pred <- predict(svm_model, newdata = X_test_scaled_inner, probability = TRUE)
        probs <- attr(pred, "probabilities")
        
        loss <- loss + log_loss(probs, y_test_inner)
      }
      if (loss < best_loss){
        best_loss <- loss
        best_gamma <- gam
      }
    }
    
    y_train <- train_data$ShotType
    y_test  <- test_data$ShotType
    
    train_features <- train_data[ , !(names(train_data) == "ShotType")]
    test_features  <- test_data[ , !(names(test_data) == "ShotType")]
    
    
    dummies_model <- dummyVars(~ ., data = train_features, fullRank = TRUE)
    
    X_train <- predict(dummies_model, newdata = train_features)
    X_test  <- predict(dummies_model, newdata = test_features, na.action = na.pass)
    
    scaler <- preProcess(X_train, method = c("center", "scale"))
    X_train_scaled <- predict(scaler, X_train)
    X_test_scaled  <- predict(scaler, X_test)
    
    svm_model <- svm(x = X_train_scaled, y = y_train, 
                     probability = TRUE, kernel = "radial", gamma=best_gamma)
    
    svm_pred <- predict(svm_model, newdata = X_test_scaled, probability = TRUE)
    svm_probs <- attr(svm_pred, "probabilities")
    
    final_loss <- final_loss + log_loss(svm_probs, y_test)
  }
  return(final_loss / num_folds)
}
rets <- SVM_CV_flat(df, fold_indices, gamma_values, 5)
rets2 <- SVM_CV_per_fold_tuning(df, fold_indices, gamma_values, 5)
rets3 <- SVM_CV_nested(df, fold_indices, gamma_values, 5)
rets3
