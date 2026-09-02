library(tidyverse)
library(caret)
library(randomForest)
library(kernelshap)

dir.create("tables", showWarnings = FALSE)
dir.create("figures", showWarnings = FALSE)
dir.create("results", showWarnings = FALSE)

set.seed(123)

train_index <- createDataPartition(
  mtcars$mpg,
  p = 0.8,
  list = FALSE
)

train_data <- mtcars[train_index, ]
test_data <- mtcars[-train_index, ]

predictors <- setdiff(names(train_data), "mpg")

tree_values <- c(50, 100, 250, 500, 1000)

runs <- 100

stability_results <- data.frame()

for (ntree in tree_values) {
  
  for (seed in 1:runs) {
    
    set.seed(seed)
    
    rf_model <- randomForest(
      mpg ~ .,
      data = train_data,
      ntree = ntree
    )
    
    shap_values <- kernelshap(
      object = rf_model,
      X = test_data[, predictors],
      bg_X = train_data[, predictors]
    )
    
    global_shap <- tibble(
      Trees = ntree,
      Run = seed,
      Feature = colnames(shap_values$S),
      Mean_Absolute_SHAP = colMeans(abs(shap_values$S))
    )
    
    stability_results <- bind_rows(
      stability_results,
      global_shap
    )
  }
}

stability_summary <-
  stability_results %>%
  group_by(Trees, Feature) %>%
  summarise(
    SD_SHAP = sd(Mean_Absolute_SHAP),
    Mean_Absolute_SHAP = mean(Mean_Absolute_SHAP),
    .groups = "drop"
  )

write.csv(
  stability_summary,
  "tables/A06_shap_tree_stability.csv",
  row.names = FALSE
)

saveRDS(
  stability_results,
  "results/A06_shap_tree_stability.rds"
)

ggplot(
  stability_results,
  aes(
    x = factor(Trees),
    y = Mean_Absolute_SHAP
  )
) +
  geom_boxplot() +
  facet_wrap(~Feature, scales = "free_y") +
  labs(
    title = "SHAP Stability Across Different Numbers of Trees",
    x = "Number of Trees",
    y = "Mean Absolute SHAP"
  ) +
  theme_minimal()

ggsave(
  "figures/A06_tree_stability_boxplot.png",
  width = 12,
  height = 8
)

sd_plot <-
  stability_summary %>%
  group_by(Trees) %>%
  summarise(
    Average_SD = mean(SD_SHAP),
    .groups = "drop"
  )

ggplot(
  sd_plot,
  aes(
    x = Trees,
    y = Average_SD
  )
) +
  geom_line() +
  geom_point(size = 2) +
  labs(
    title = "Average SHAP Stability Across Tree Numbers",
    x = "Number of Trees",
    y = "Average Standard Deviation"
  ) +
  theme_minimal()

ggsave(
  "figures/A06_tree_stability_sd.png",
  width = 8,
  height = 6
)

print(stability_summary, n = 50)