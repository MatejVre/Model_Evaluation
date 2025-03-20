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
#Correlation shows that there is a negative relationship between distance and log loss

SVM_nested_df <- data.frame(
  Distance = ordered_distances,
  Evals = evals_SVM_nested
)

SVM_nested_df$Distance_rounded <- round(SVM_nested_df$Distance)

test_df <- SVM_nested_df %>% group_by(Distance_rounded)%>%
  summarise(mean_accuracy = mean(Evals.acc_error_vec), se = sd(Evals.acc_error_vec) / sqrt(n()))

ggplot(test_df, aes(x=Distance_rounded, y=mean_accuracy)) +
  geom_bar(stat="identity", fill="steelblue") + 
  geom_errorbar(aes(ymin = mean_accuracy - se, ymax = mean_accuracy + se), width = 0.2, color="black") +
  labs(x = "Rounded Distance", y = "Average Accuracy")

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
  
  
  