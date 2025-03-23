library("dplyr")
library("VGAM")
library("nnet")
library(caret)
library(class)
library(e1071)
library(glue)
library(rpart)

df <- read.csv("dataset.csv", sep=";", header=TRUE)

X <- df[, !(names(df) %in% c("ShotType"))]
y <- df["ShotType"]

df %>% group_by(df$ShotType) %>% summarize(count=n())
#above head appears 3055 times and it is most frequent
#tip-in appears 61 times and it is the least frequent
#Try stratified CV to keep the distributions of shot types

#LOG-LOSS
log_loss <- function(probabilities, y_true) {
  N <- length(y_true)
  losses <- numeric(N)
  
  for (i in 1:N) {
    p <- probabilities[i, as.character(y_true[i])]
    p <- max(p, 1e-15)  # avoid log(0)
    losses[i] <- -log(p)
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

stratified_folds <- function(data, target_column, k=10){
  set.seed(42)#remove if doing repeated CV!
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
    #df$ShotType <- as.factor(df$ShotType)
    
    test_indices <- fold_indices[[i]]
    
    train_data <- df[-test_indices,]
    test_data <- df[test_indices,]
    
    model <- multinom(ShotType ~ .,data = train_data, trace = FALSE)
    
    prediction_probabilities <- predict(model, newdata = test_data, type = "probs")
    prediction_labels <- predict(model, newdata = test_data, type = "class")
    
    loss_list <- log_loss(prediction_probabilities, test_data$ShotType)
    acc_list <- accuracy(prediction_labels, test_data$ShotType)
    
    losses <- c(losses, loss_list[["log_loss"]])
    accs <- c(accs, acc_list[["accuracy"]])
    loss_vector <- c(loss_vector, loss_list[["loss_vector"]])
    err_vector <- c(err_vector, acc_list[["err_vector"]])
  }
  
  evals <- list(log_loss = mean(losses), loss_vector = loss_vector, accuracy = mean(accs), acc_error_vec = err_vector)
  return(evals)
}

depths <- c(5, 10, 15, 20, 25,30)

CT_CV_per_fold_tuning <- function(df, fold_indices, depths){
  
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
    
    best_depth <- NA
    best_fold_loss <- Inf
    
    for (depth in depths) {
      tree_model <- rpart(ShotType ~ ., data = train_data, method="class",
                          control = rpart.control(maxdepth=depth, cp=0))
      
      pred <- predict(tree_model, newdata = train_data, type="class")
      probs <- predict(tree_model, newdata = train_data, type="prob")
      
      fold_loss <- log_loss(probs, train_data$ShotType)[[1]]
      #print(accuracy(pred, train_data$ShotType)[["accuracy"]])
      
      if (fold_loss < best_fold_loss) {
        best_fold_loss <- fold_loss
        best_depth <- depth
      }
    }
    print(best_depth)
    final_model <- rpart(ShotType ~ ., data = train_data, method="class",
                         control = rpart.control(maxdepth=best_depth, cp=0))
    
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

CT_CV_nested <- function(df, fold_indices, depths){
  
  losses <- c()
  loss_vector <- c()
  accs <- c()
  err_vector <- c()
  
  df$ShotType <- as.factor(df$ShotType)
  best_depth <- NA
  for (i in 1:length(fold_indices)){
    
    best_loss <- Inf
    test_indices <- fold_indices[[i]]
    train_data <- df[-test_indices, ]
    test_data  <- df[test_indices, ]
    
    for (depth in depths){
      
      loss <- 0
      
      inner_fold_indices <- stratified_folds(train_data, "ShotType", 7)
      for (u in 1:length(inner_fold_indices)){
        
        inner_test_indices <- inner_fold_indices[[u]]
        inner_train_data <- train_data[-inner_test_indices, ]
        inner_test_data  <- train_data[inner_test_indices, ]
        
        y_train_inner <- inner_train_data$ShotType
        y_test_inner  <- inner_test_data$ShotType
        
        tree_model <- rpart(ShotType ~ ., data = inner_train_data, method="class",
                            control = rpart.control(maxdepth=depth, cp=0))
        
        pred <- predict(tree_model, newdata = inner_test_data, type="class")
        probs <- predict(tree_model, newdata = inner_test_data, type="prob")
        
        loss <- loss + log_loss(probs, y_test_inner)[[1]]
      }
      if (loss < best_loss){
        best_loss <- loss
        best_depth <- depth
      }
    }
    print(best_depth)
    tree_model <- rpart(ShotType ~ ., data = train_data, method="class",
                        control = rpart.control(maxdepth=best_depth, cp=0))
    
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
  set.seed(42)
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
       " +/- ",
       bootstrap_uncertainty(evals_list[["acc_error_vec"]]))
}

set.seed(42)
num_folds <- 10
fold_indices <- stratified_folds(df, "ShotType", k=num_folds)

evals_baseline <- baseline_CV_evaluation(df, fold_indices)
evals_LR <- LR_CV_evaluation(df, fold_indices)
evals_tree_training_fold <- CT_CV_per_fold_tuning(df, fold_indices, depths)
evals_tree_nested <- CT_CV_nested(df, fold_indices, depths)

report_metrics(evals_baseline)
report_metrics(evals_LR)
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

