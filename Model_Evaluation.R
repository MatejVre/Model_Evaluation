library("dplyr")
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
    loss <- loss - (1/N) * log(probabilities[i, y_true[i]])
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

