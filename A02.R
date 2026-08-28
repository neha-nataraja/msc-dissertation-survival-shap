# Load Packages

library(tidyverse)
library(caret)
library(randomForest)
library(kernelshap)
library(ggplot2)

# Load Dataset

data("mtcars")

# Data Preparation

set.seed(123)

train_index <- createDataPartition(mtcars$mpg, p = 0.8, list = FALSE)

train_data <- mtcars[train_index, ]
test_data <- mtcars[-train_index, ]

# Fit Random Forest Model

rf_model <- randomForest(
  mpg ~ .,
  data = train_data,
  ntree = 500,
  importance = TRUE
)

# Select Observation

new_observation <- test_data[1, ]

# Calculate SHAP Values

shap_values <- kernelshap(
  object = rf_model,
  X = new_observation,
  bg_X = train_data
)

# Extract SHAP Values

shap_results <- data.frame(
  Feature = colnames(shap_values$S),
  SHAP = as.numeric(shap_values$S[1, ])
)

print(shap_results)

# Save Results

write.csv(
  shap_results,
  "tables/A02_shap_values.csv",
  row.names = FALSE
)

saveRDS(
  shap_values,
  "results/A02_shap_values.rds"
)

# Calculate Baseline and Prediction

baseline <- shap_values$baseline

prediction <- predict(rf_model, new_observation)

shap_sum <- sum(shap_results$SHAP)

efficiency_results <- data.frame(
  Baseline = baseline,
  SHAP_Sum = shap_sum,
  Baseline_Plus_SHAP = baseline + shap_sum,
  Model_Prediction = prediction
)

print(efficiency_results)

# Save Efficiency Results

write.csv(
  efficiency_results,
  "tables/A02_efficiency_validation.csv",
  row.names = FALSE
)

# SHAP Efficiency Plot

efficiency_plot <- ggplot(
  shap_results,
  aes(
    x = reorder(Feature, SHAP),
    y = SHAP
  )
) +
  geom_col() +
  coord_flip() +
  labs(
    title = "SHAP Feature Contributions",
    x = "Feature",
    y = "SHAP Value"
  ) +
  theme_minimal()

ggsave(
  filename = "figures/A02_shap_efficiency.png",
  plot = efficiency_plot,
  width = 7,
  height = 5,
  dpi = 300
)

