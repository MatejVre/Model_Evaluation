library(ggplot2)
library(gridExtra)
indices <- c()
################################################################################
#ANALYSIS OF ERRORS BASED ON DISTANCE
################################################################################
indices <- c()#if this isn't set before the for loop is ran everything breaks
for (indice_vec in fold_indices){
  indices <- c(indices, indice_vec)
}

ordered_distances <- df[indices, "Distance"]

correlation_bootstrap <- function(distances, evals_list, n=1000){
  set.seed(42)#set seed so all 4 function calls work with the same samples
  corrs <- c()
  for (i in 1:n){
    indices <- sample(1:length(distances), length(distances), replace=TRUE)
    d <- distances[c(indices)]
    errs <- evals_list[["loss_vector"]][c(indices)]
    corrs <- c(corrs, cor(d, errs))
  }
  return(sd(corrs))
}
report_correlation <- function(distances, evals_list, name){
  glue(
    name,
    " correlation with distance: ",
    cor(distances, evals_list[["loss_vector"]]),
    " +/- ",
    correlation_bootstrap(distances, evals_list)
  )
}

report_correlation(ordered_distances, evals_baseline, "Baseline")
report_correlation(ordered_distances, evals_LR, "Logistic Regression")
report_correlation(ordered_distances, evals_tree_training_fold, "Tree - optimized training fold")
report_correlation(ordered_distances, evals_tree_nested, "Tree - nested CV")

create_loss_plot <- function(evals_list, title, color){
  tree_nested_df <- data.frame(
    Distance = ordered_distances,
    Evals = evals_list
  )
  
  tree_nested_df$Distance_rounded <- round(tree_nested_df$Distance)
  
  test_df <- tree_nested_df %>% group_by(Distance_rounded)%>%
    summarise(mean_loss = mean(Evals.loss_vector), se = sd(Evals.loss_vector) / sqrt(n()))
  
  ggplot(test_df, aes(x=Distance_rounded, y=mean_loss)) +
    geom_bar(stat="identity", fill=color) + 
    geom_errorbar(aes(ymin = mean_loss - se, ymax = mean_loss + se), width = 0.2, color="black") +
    labs(x = "Rounded Distance", y = "Average Log Loss", title=title)
}

p1 <- create_loss_plot(evals_baseline, "Baseline", "#1b9e77")
p2 <- create_loss_plot(evals_LR, "Logistic Regression", "#d95f02")
p3 <- create_loss_plot(evals_tree_training_fold, "Tree - Optimized Training Fold", "#7570b3")
p4 <- create_loss_plot(evals_tree_nested, "Tree - Nested Cross Validation", "#66a61e")

grid.arrange(p1, p2, p3, p4, nrow = 2, ncol = 2)

#It is evident that the further away we are from the basket, the higher the accuracy. The shots at 9 and
#10 meters away from the basket show a downwards trend, however, their respective errors are also much higher,
#meaning we can most likely ignore their findings since there aren't many shots recorded for those distances.

far_away_shots_df <- df %>% filter(Distance >= 5)

df$Group <- "All Shots"
far_away_shots_df$Group <- "Further than 5m"

combined_df <- rbind(df, far_away_shots_df)

normalized_df <- combined_df %>%
  group_by(Group, ShotType) %>%
  summarise(count = n(), .groups = "drop") %>%
  group_by(Group) %>%
  mutate(prop = count / sum(count))

ggplot(normalized_df, aes(x = ShotType, y = prop, fill = Group)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "Shot Type", y = "Proportion"
  ) +
  scale_y_continuous(labels = scales::percent_format()) +
  theme(legend.position = c(0.9,0.9))

################################################################################
#CORRECTLY WEIGHING THE DATA
################################################################################
#look at the piece of paper for proof
weighted_df <- read.csv("dataset.csv", sep=";", header=TRUE)
weighted_df <- weighted_df[indices, ]
weighted_df$Weight <- weights[weighted_df$Competition]/data_weights[weighted_df$Competition]

get_weighted_loss <- function(df, evals_list){
  temp <- df
  temp$Loss <- evals_list[["loss_vector"]]
  
  weighted_avg_loss <- sum(temp$Weight * temp$Loss) / sum(temp$Weight)
  return(weighted_avg_loss)
}

get_weighted_accuracy <- function(df, evals_list){
  temp <- df
  temp$Accuracy <- evals_list[["acc_error_vec"]]
  
  weighted_avg_accuracy <- sum(temp$Weight * temp$Accuracy) / sum(temp$Weight)
  return(weighted_avg_accuracy)
}

get_weighted_loss(weighted_df, evals_baseline)
get_weighted_loss(weighted_df, evals_LR)
get_weighted_loss(weighted_df, evals_tree_training_fold)
get_weighted_loss(weighted_df, evals_tree_nested)

get_weighted_accuracy(weighted_df, evals_baseline)
get_weighted_accuracy(weighted_df, evals_LR)
get_weighted_accuracy(weighted_df, evals_tree_training_fold)
get_weighted_accuracy(weighted_df, evals_tree_nested)

test <- weighted_df
test$Loss <- evals_tree_training_fold[["loss_vector"]]
test$WeightedError <- test$Weight*test$Loss
bootstrap_uncertainty(test$WeightedError)

#About bootstrap. There are two ways to think about it, do normal bootstrap, which will likely return
#higher uncertainty. However, weighted bootstrap should return smaller uncertainty. Should the uncertainty
#for "artificially" changing the distribution be the same as the original one, or should it be higher?
#Who knows





