# Load Packages

library(tidyverse)
library(caret)
library(randomForest)
library(kernelshap)
library(ggplot2)

# Load Dataset

data("mtcars")

# Define Random Seeds

seeds <- c(
  123, 456, 789, 1011, 2022,
  3033, 4044, 5055, 6066, 7077,
  8088, 9099, 1111, 2222, 3333,
  4444, 5555, 6666, 7777, 8888
)

# Initialise Results

stability_results <- data.frame()

# Create Train/Test Split

set.seed(123)

train_index <- createDataPartition(
  mtcars$mpg,
  p = 0.8,
  list = FALSE
)

train_data <- mtcars[train_index, ]
test_data <- mtcars[-train_index, ]

predictors <- setdiff(names(train_data), "mpg")

# Calculate SHAP Stability

for (seed in seeds) {
  
  set.seed(seed)
  
  rf_model <- randomForest(
    mpg ~ .,
    data = train_data,
    ntree = 500,
    importance = TRUE
  )
  
  shap_values <- kernelshap(
    object = rf_model,
    X = test_data[, predictors],
    bg_X = train_data[, predictors]
  )
  
  global_shap <- data.frame(
    Feature = colnames(shap_values$S),
    Mean_Absolute_SHAP = colMeans(abs(shap_values$S))
  )
  
  global_shap$Seed <- seed
  
  stability_results <- bind_rows(
    stability_results,
    global_shap
  )
}

# Calculate Stability Summary

stability_summary <- stability_results %>%
  group_by(Feature) %>%
  summarise(
    Mean_SHAP = mean(Mean_Absolute_SHAP),
    SD_SHAP = sd(Mean_Absolute_SHAP),
    .groups = "drop"
  ) %>%
  arrange(desc(Mean_SHAP))

print(stability_summary)

# Stability Plot

stability_plot <- ggplot(
  stability_summary,
  aes(
    x = reorder(Feature, Mean_SHAP),
    y = Mean_SHAP
  )
) +
  geom_col(fill = "steelblue") +
  geom_errorbar(
    aes(
      ymin = Mean_SHAP - SD_SHAP,
      ymax = Mean_SHAP + SD_SHAP
    ),
    width = 0.2
  ) +
  coord_flip() +
  labs(
    title = "SHAP Stability Across Random Seeds",
    x = "Feature",
    y = "Mean Absolute SHAP"
  ) +
  theme_minimal()

ggsave(
  filename = "figures/A05_shap_stability.png",
  plot = stability_plot,
  width = 7,
  height = 5,
  dpi = 300
)

# Save Stability Results

write.csv(
  stability_summary,
  "tables/A05_shap_stability.csv",
  row.names = FALSE
)