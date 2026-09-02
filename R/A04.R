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

train_index <- createDataPartition(
  mtcars$mpg,
  p = 0.8,
  list = FALSE
)

train_data <- mtcars[train_index, ]
test_data <- mtcars[-train_index, ]

# Fit Random Forest Model

rf_model <- randomForest(
  mpg ~ .,
  data = train_data,
  ntree = 500,
  importance = TRUE
)

# Select Predictor Variables

predictors <- setdiff(names(train_data), "mpg")

# Calculate SHAP Values

shap_values <- kernelshap(
  object = rf_model,
  X = test_data[, predictors],
  bg_X = train_data[, predictors]
)

# Global Feature Importance

global_shap <- data.frame(
  Feature = colnames(shap_values$S),
  Mean_Absolute_SHAP = colMeans(abs(shap_values$S))
)

global_shap <- global_shap %>%
  arrange(desc(Mean_Absolute_SHAP))

print(global_shap)

# Global SHAP Plot

global_shap_plot <- ggplot(
  global_shap,
  aes(
    x = reorder(Feature, Mean_Absolute_SHAP),
    y = Mean_Absolute_SHAP
  )
) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(
    title = "Global Feature Importance using Mean Absolute SHAP",
    x = "Feature",
    y = "Mean Absolute SHAP"
  ) +
  theme_minimal()

ggsave(
  filename = "figures/A04_global_shap.png",
  plot = global_shap_plot,
  width = 7,
  height = 5,
  dpi = 300
)

# Save Global SHAP Results

write.csv(
  global_shap,
  "tables/A04_global_shap.csv",
  row.names = FALSE
)

# Compare Mean SHAP and Mean Absolute SHAP

shap_summary <- data.frame(
  Feature = colnames(shap_values$S),
  Mean_SHAP = colMeans(shap_values$S),
  Mean_Absolute_SHAP = colMeans(abs(shap_values$S))
)

shap_summary <- shap_summary %>%
  arrange(desc(Mean_Absolute_SHAP))

print(shap_summary)

# Save SHAP Summary

write.csv(
  shap_summary,
  "tables/A04_shap_summary.csv",
  row.names = FALSE
)