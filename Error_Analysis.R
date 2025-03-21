library(ggplot2)
library(gridExtra)
indices <- c()

for (indice_vec in fold_indices){
  indices <- c(indices, indice_vec)
}

ordered_distances <- df[indices, "Distance"]

cor(ordered_distances, evals_baseline[["loss_vector"]])
cor(ordered_distances, evals_LR[["loss_vector"]])
cor(ordered_distances, evals_tree_training_fold[["loss_vector"]])
cor(ordered_distances, evals_tree_nested[["loss_vector"]])
#Correlation shows that there is a negative relationship between distance and log loss
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
far_away_shots_df$Group <- "Far Away"

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
  scale_y_continuous(labels = scales::percent_format())
  

data_weights <- table(df$Competition)/nrow(df)

weights <- c("NBA"= 0.6, "U14"=0.1, "U16"=0.1, "SLO1"=0.1, "EURO"=0.1)
data_weights["NBA"]
weights["NBA"]/data_weights["NBA"]

