library("dplyr")
library("VGAM")
library("nnet")
library(caret)
library(class)
library(e1071)
library(glue)
library(rpart)
set.seed(42)
df <- read.csv("dataset.csv", sep=";", header=TRUE)
X <- df[, !(names(df) %in% c("ShotType"))]
y <- df["ShotType"]

df %>% group_by(df$ShotType) %>% summarize(count=n())
#above head appears 3055 times and it is most frequent
#tip-in appears 61 times and it is the least frequent
#Try stratified CV to keep the distributions of shot types

#Questions to ask:
#Should baseline always predict the max class or sample form learned distribution.
#Bootstrap the whole error vector or all k of them individually and average?
# TODO b/a weighs

#LOG-LOSS
log_loss <- function(probabilities, y_true){
  losses <- c()
  N <-  length(y_true)
  for (i in 1:N){
    probability <- probabilities[i, as.character(y_true[i])]
    probability <- max(probability, 1e-15)
    loss <- loss - (1/N) * log(probability)
    losses <- c(losses, loss)
  }
  return(list(log_loss = mean(losses), loss_vector = losses))
}

#ACCURACY
accuracy <- function(predictions, y_true){
  accs <- (predictions == y_true)
  return(list(accuracy = mean(accs), err_vector = accs))
}


train_baseline_classifier <- function(y_train){
  return(prop.table(table(y_train)))
}

#USE TO PREDICT LABELS - ACCURACY
probabilities <- train_baseline_classifier(y)
predict_baseline_classifier <- function(probabilities, n=1){
  return(rep(tail(names(sort(probabilities)), 1),n))
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

#Potentially useless
test_train_split <- function(data, train_size = 0.8){
  train_indices <- sample(1:nrow(data), size = floor(train_size * nrow(data)))
  
  train_set <- data[train_indices, ]
  test_set <- data[-train_indices,]
  return(list(train = train_set, test = test_set))
}

baseline_CV_evaluation <- function(df, fold_indices){
  
  losses <- c()
  loss_vector <- c()
  accs <- c()
  err_vector <- c()
  
  for (i in 1:length(fold_indices)){
    df$ShotType <- as.factor(df$ShotType)
    
    test_indices <- fold_indices[[i]]
    train_data <- df[-test_indices,]
    test_data <- df[test_indices,]
    
    baseline_probs <- train_baseline_classifier(train_data$ShotType)
    baseline_prediction_probabilities <- baseline_classifier_probabilities(baseline_probs, nrow(test_data))
    baseline_prediction_labels <- predict_baseline_classifier(baseline_probs, nrow(test_data))
    
    loss_list <- log_loss(baseline_prediction_probabilities, test_data$ShotType)
    acc_list <- accuracy(baseline_prediction_labels, test_data$ShotType)
    
    losses <- c(losses, loss_list[[1]])
    accs <- c(accs, acc_list[[1]])
    loss_vector <- c(loss_vector, loss_list[[2]])
    err_vector <- c(err_vector, acc_list[[2]])
  }
  
  evals <- list(log_loss = mean(losses), loss_vector = loss_vector, accuracy = mean(accs), acc_error_vec = err_vector)
  return(evals)
}

LR_CV_evaluation <- function(df, fold_indices){
  
  losses <- c()
  loss_vector <- c()
  accs <- c()
  err_vector <- c()
  
  for (i in 1:length(fold_indices)){
    df$ShotType <- as.factor(df$ShotType)
    
    test_indices <- fold_indices[[i]]
    
    train_data <- df[-test_indices,]
    test_data <- df[test_indices,]
    
    model <- multinom(ShotType ~ ., family = multinomial, data = train_data, trace = FALSE)
    
    prediction_probabilities <- predict(model, newdata = test_data, type = "probs")
    prediction_labels <- predict(model, newdata = test_data, type = "class")
    
    loss_list <- log_loss(prediction_probabilities, test_data$ShotType)
    acc_list <- accuracy(prediction_labels, test_data$ShotType)
    
    losses <- c(losses, loss_list[[1]])
    accs <- c(accs, acc_list[[1]])
    loss_vector <- c(loss_vector, loss_list[[2]])
    err_vector <- c(err_vector, acc_list[[2]])
  }
  
  evals <- list(log_loss = mean(losses), loss_vector = loss_vector, accuracy = mean(accs), acc_error_vec = err_vector)
  return(evals)
}

gamma_values <- c(0.001, 0.01, 0.1, 1, 10, 100)
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
SVM_CV_per_fold_tuning <- function(df, fold_indices, gamma_values) {
  
  losses <- c()
  loss_vector <- c()
  accs <- c()
  err_vector <- c()
  
  df$ShotType <- as.factor(df$ShotType)
  
  for (i in 1:length(fold_indices)) {
    
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

      fold_loss <- log_loss(probs, y_train)[[1]]
      
      if (fold_loss < best_fold_loss) {
        best_fold_loss <- fold_loss
        best_gamma <- gam
      }
    }
    
    final_model <- svm(x = X_train_scaled, y = y_train,
                              probability = TRUE, kernel = "radial", gamma = best_gamma)
    
    final_preds <- predict(final_model, newdata = X_test_scaled, probability = TRUE)
    final_probs <- attr(final_preds, "probabilities")
    
    loss_list <- log_loss(final_probs, y_test)
    acc_list <- accuracy(final_preds, y_test)
    
    losses <- c(losses, loss_list[[1]])
    accs <- c(accs, acc_list[[1]])
    loss_vector <- c(loss_vector, loss_list[[2]])
    err_vector <- c(err_vector, acc_list[[2]])
  }
  
  evals <- list(log_loss = mean(losses), loss_vector = loss_vector, accuracy = mean(accs), acc_error_vec = err_vector)
  return(evals)
}

SVM_CV_nested <- function(df, fold_indices, gamma_values) {
  
  losses <- c()
  loss_vector <- c()
  accs <- c()
  err_vector <- c()
  
  df$ShotType <- as.factor(df$ShotType)
  best_gamma <- NA
  for (i in 1:length(fold_indices)){
    
    best_loss <- Inf
    test_indices <- fold_indices[[i]]
    train_data <- df[-test_indices, ]
    test_data  <- df[test_indices, ]
    
    for (gam in gamma_values){
      
      loss <- 0
      
      inner_fold_indices <- stratified_folds(train_data, "ShotType", 3)
      for (u in 1:length(inner_fold_indices)){
        
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
        
        loss <- loss + log_loss(probs, y_test_inner)[[1]]
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
    
    loss_list <- log_loss(svm_probs, test_data$ShotType)
    acc_list <- accuracy(svm_pred, test_data$ShotType)
    
    losses <- c(losses, loss_list[[1]])
    accs <- c(accs, acc_list[[1]])
    loss_vector <- c(loss_vector, loss_list[[2]])
    err_vector <- c(err_vector, acc_list[[2]])
  }
  
  evals <- list(log_loss = mean(losses), loss_vector = loss_vector, accuracy = mean(accs), acc_error_vec = err_vector)
  return(evals)
}
#rets <- SVM_CV_flat(df, fold_indices, gamma_values)#arguably the wrong version
#rets2 <- SVM_CV_per_fold_tuning(df, fold_indices, gamma_values)
#rets3 <- SVM_CV_nested(df, fold_indices, gamma_values, 5)
cps <- c(0.0001, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1)
depths <- c(5, 10, 15, 20, 25,30)
CT_CV_per_fold_tuning <- function(df, fold_indices, cps){
  
  losses <- c()
  loss_vector <- c()
  accs <- c()
  err_vector <- c()
  
  df$ShotType <- as.factor(df$ShotType)
  
  for (i in 1:length(fold_indices)) {
    
    test_indices <- fold_indices[[i]]
    train_data <- df[-test_indices, ]
    test_data  <- df[test_indices, ]
    
    y_test  <- test_data$ShotType
    
    best_cp <- NA
    best_fold_loss <- Inf
    
    for (cp in cps) {
      tree_model <- rpart(ShotType ~ ., data = train_data, method="class",
                          control = rpart.control(maxdepth=cp, cp=0))
      
      pred <- predict(tree_model, newdata = train_data, type="class")
      probs <- predict(tree_model, newdata = train_data, type="prob")
      
      fold_loss <- log_loss(probs, train_data$ShotType)[[1]]
      
      if (fold_loss < best_fold_loss) {
        best_fold_loss <- fold_loss
        best_cp <- cp
      }
    }
    
    final_model <- rpart(ShotType ~ ., data = train_data, method="class",
                         control = rpart.control(maxdepth=best_cp, cp=0))
    
    final_preds <- predict(final_model, newdata = test_data, type="class")
    final_probs <- predict(final_model, newdata = test_data, type="prob")
    
    loss_list <- log_loss(final_probs, y_test)
    acc_list <- accuracy(final_preds, y_test)
    
    losses <- c(losses, loss_list[[1]])
    accs <- c(accs, acc_list[[1]])
    loss_vector <- c(loss_vector, loss_list[[2]])
    err_vector <- c(err_vector, acc_list[[2]])
  }
  
  evals <- list(log_loss = mean(losses), loss_vector = loss_vector, accuracy = mean(accs), acc_error_vec = err_vector)
  return(evals)
}


CT_CV_nested <- function(df, fold_indices, cps){
  
  losses <- c()
  loss_vector <- c()
  accs <- c()
  err_vector <- c()
  
  df$ShotType <- as.factor(df$ShotType)
  best_cp <- NA
  for (i in 1:length(fold_indices)){
    
    best_loss <- Inf
    test_indices <- fold_indices[[i]]
    train_data <- df[-test_indices, ]
    test_data  <- df[test_indices, ]
    
    for (cp in cps){
      
      loss <- 0
      
      inner_fold_indices <- stratified_folds(train_data, "ShotType", 3)
      for (u in 1:length(inner_fold_indices)){
        
        inner_test_indices <- inner_fold_indices[[u]]
        inner_train_data <- train_data[-inner_test_indices, ]
        inner_test_data  <- train_data[inner_test_indices, ]
        
        y_train_inner <- inner_train_data$ShotType
        y_test_inner  <- inner_test_data$ShotType
        
        tree_model <- rpart(ShotType ~ ., data = inner_train_data, method="class",
                            control = rpart.control(maxdepth=cp, cp=0))
        
        pred <- predict(tree_model, newdata = inner_test_data, type="class")
        probs <- predict(tree_model, newdata = inner_test_data, type="prob")
        
        loss <- loss + log_loss(probs, y_test_inner)[[1]]
      }
      if (loss < best_loss){
        best_loss <- loss
        best_cp <- cp
      }
    }
    
    tree_model <- rpart(ShotType ~ ., data = train_data, method="class",
                        control = rpart.control(maxdepth=best_cp, cp=0))
    
    tree_pred <- predict(tree_model, newdata = test_data, type="class")
    tree_probs <- predict(tree_model, newdata = test_data, type="prob")
    
    loss_list <- log_loss(tree_probs, test_data$ShotType)
    acc_list <- accuracy(tree_pred, test_data$ShotType)
    
    losses <- c(losses, loss_list[[1]])
    accs <- c(accs, acc_list[[1]])
    loss_vector <- c(loss_vector, loss_list[[2]])
    err_vector <- c(err_vector, acc_list[[2]])
  }
  
  evals <- list(log_loss = mean(losses), loss_vector = loss_vector, accuracy = mean(accs), acc_error_vec = err_vector)
  return(evals)
}  


bootstrap_uncertainty <- function(error_vector){
  means <- c()
  for (i in 1:1000){
    bootstrap_sample <- sample(error_vector, length(error_vector), replace = TRUE)
    bootstrap_mean <- mean(bootstrap_sample)
    means <- c(means, bootstrap_mean)
  }
  return(sd(means))
}

report_metrics <- function(evals_list){
  
  glue("Log loss: ", evals_list[["log_loss"]],
       " +/- ",
       bootstrap_uncertainty(evals_list[["loss_vector"]]),
       "\n",
       "Accuracy: ", evals_list[["accuracy"]],
       " +/ -",
       bootstrap_uncertainty(evals_list[["acc_error_vec"]]))
}

set.seed(42)
num_folds <- 5
fold_indices <- stratified_folds(df, "ShotType", k=num_folds)

evals_baseline <- baseline_CV_evaluation(df, fold_indices)
evals_LR <- LR_CV_evaluation(df, fold_indices)
#evals_SVM_training_fold <- SVM_CV_per_fold_tuning(df, fold_indices, gamma_values)
#evals_SVM_nested <- SVM_CV_nested(df, fold_indices, gamma_values)
evals_tree_training_fold <- CT_CV_per_fold_tuning(df, fold_indices, depths)
evals_tree_nested <- CT_CV_nested(df, fold_indices, depths)

report_metrics(evals_baseline)
report_metrics(evals_LR)
#report_metrics(evals_SVM_training_fold)
report_metrics(evals_tree_training_fold)
report_metrics(evals_tree_nested)

#NOTES
#test: m
#train: n - m
#big m means high bias
#different parameters for each fold is OK!
#Choosing K - 
#LOO? Goes through all possible models
#10 repetitions of 2 cross validation alternative to LOO

