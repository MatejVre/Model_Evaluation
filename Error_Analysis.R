library(ggplot2)

indices <- c()

for (indice_vec in fold_indices){
  indices <- c(indices, indice_vec)
}

ordered_distances <- df[indices, "Distance"]

cor(ordered_distances, evals_baseline[["loss_vector"]])
cor(ordered_distances, evals_LR[["loss_vector"]])
cor(ordered_distances, evals_SVM_training_fold[["loss_vector"]])
cor(ordered_distances, evals_SVM_nested[["loss_vector"]])

