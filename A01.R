# Load Packages

library(tidyverse)
library(caret)
library(randomForest)
library(Metrics)

# Load Dataset

data("mtcars")

# Data Preparation

set.seed(123)

train_index <- createDataPartition(mtcars$mpg, p = 0.8, list = FALSE)

train_data <- mtcars[train_index, ]
test_data <- mtcars[-train_index, ]

# Fit Linear Regression Model

lm_model <- lm(mpg ~ ., data = train_data)

# Fit Random Forest Model

rf_model <- randomForest(
  mpg ~ .,
  data = train_data,
  ntree = 500,
  importance = TRUE
)

# Make Predictions

lm_predictions <- predict(lm_model, newdata = test_data)
rf_predictions <- predict(rf_model, newdata = test_data)

# Model Evaluation

lm_rmse <- rmse(test_data$mpg, lm_predictions)
rf_rmse <- rmse(test_data$mpg, rf_predictions)

lm_mae <- mae(test_data$mpg, lm_predictions)
rf_mae <- mae(test_data$mpg, rf_predictions)

lm_r2 <- cor(test_data$mpg, lm_predictions)^2
rf_r2 <- cor(test_data$mpg, rf_predictions)^2

model_metrics <- data.frame(
  Model = c("Linear Regression", "Random Forest"),
  RMSE = c(lm_rmse, rf_rmse),
  MAE = c(lm_mae, rf_mae),
  R2 = c(lm_r2, rf_r2)
)

print(model_metrics)

# Save Results

write.csv(
  model_metrics,
  "tables/A01_model_metrics.csv",
  row.names = FALSE
)

saveRDS(
  list(
    lm_model = lm_model,
    rf_model = rf_model
  ),
  "results/A01_models.rds"
)

# Observed vs Predicted - Linear Regression

lm_plot <- ggplot(
  data.frame(
    Actual = test_data$mpg,
    Predicted = lm_predictions
  ),
  aes(x = Actual, y = Predicted)
) +
  geom_point(size = 3) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  labs(
    title = "Observed vs Predicted: Linear Regression",
    x = "Actual MPG",
    y = "Predicted MPG"
  ) +
  theme_minimal()

ggsave(
  filename = "figures/A01_observed_vs_predicted_lm.png",
  plot = lm_plot,
  width = 6,
  height = 5,
  dpi = 300
)

# Observed vs Predicted - Random Forest

rf_plot <- ggplot(
  data.frame(
    Actual = test_data$mpg,
    Predicted = rf_predictions
  ),
  aes(x = Actual, y = Predicted)
) +
  geom_point(size = 3) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  labs(
    title = "Observed vs Predicted: Random Forest",
    x = "Actual MPG",
    y = "Predicted MPG"
  ) +
  theme_minimal()

ggsave(
  filename = "figures/A01_observed_vs_predicted_rf.png",
  plot = rf_plot,
  width = 6,
  height = 5,
  dpi = 300
)

# RMSE Comparison

rmse_plot <- ggplot(
  data.frame(
    Model = c("Linear Regression", "Random Forest"),
    RMSE = c(lm_rmse, rf_rmse)
  ),
  aes(x = Model, y = RMSE)
) +
  geom_col(width = 0.6) +
  labs(
    title = "RMSE Comparison of Regression Models",
    x = "Model",
    y = "RMSE"
  ) +
  theme_minimal()

ggsave(
  filename = "figures/A01_rmse_comparison.png",
  plot = rmse_plot,
  width = 6,
  height = 5,
  dpi = 300
)