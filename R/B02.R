# ============================================================
# B02: Insurance Model Comparison
# Two-Part GLM vs Two-Part Random Forest
# ============================================================

library(tidyverse)
library(caret)
library(randomForest)

# ------------------------------------------------------------
# 1. Load Data
# ------------------------------------------------------------

insurance_data <- readRDS("results/insurance_data.rds")
severity_data  <- readRDS("results/severity_data.rds")

# ------------------------------------------------------------
# 2. Define Predictors
# ------------------------------------------------------------

occurrence_predictors <- setdiff(
  names(insurance_data),
  c("clm", "claimcst0", "numclaims")
)

# clm is excluded here because severity_data is already filtered to
# clm == 1, making it a constant column within this subset (zero
# variance -> rank-deficient fit if left in).

severity_predictors <- setdiff(
  names(severity_data),
  c("claimcst0", "clm")
)

occurrence_formula <- as.formula(
  paste("clm ~", paste(occurrence_predictors, collapse = " + "))
)

severity_formula <- as.formula(
  paste("claimcst0 ~", paste(severity_predictors, collapse = " + "))
)

# ------------------------------------------------------------
# 3. Train-Test Split
# ------------------------------------------------------------

set.seed(123)

train_index <- createDataPartition(
  insurance_data$clm,
  p = 0.8,
  list = FALSE
)

train_data <- insurance_data[train_index, ]
test_data  <- insurance_data[-train_index, ]

severity_train <- train_data %>% filter(clm == 1)
severity_test  <- test_data %>% filter(clm == 1)

cat("Training observations:", nrow(train_data), "\n")
cat("Testing observations:", nrow(test_data), "\n")
cat("Positive claims in training:", nrow(severity_train), "\n")
cat("Positive claims in testing:", nrow(severity_test), "\n")

# ------------------------------------------------------------
# 4. TWO-PART GLM
# ------------------------------------------------------------

# 4.1 Claim Occurrence GLM

occurrence_glm <- glm(
  occurrence_formula,
  data = train_data,
  family = binomial(link = "logit")
)

glm_occurrence_probability <- predict(
  occurrence_glm,
  newdata = test_data,
  type = "response"
)

# 4.2 Claim Severity GLM

severity_glm <- glm(
  severity_formula,
  data = severity_train,
  family = Gamma(link = "log")
)

glm_severity_prediction <- predict(
  severity_glm,
  newdata = test_data,
  type = "response"
)

glm_expected_cost <- glm_occurrence_probability * glm_severity_prediction

# ------------------------------------------------------------
# 5. TWO-PART RANDOM FOREST
# ------------------------------------------------------------

# 5.1 Claim Occurrence Random Forest
# clm must be a factor here, or randomForest silently fits regression
# instead of classification and predict(..., type = "prob") fails.
# This factor conversion is scoped to *_rf copies only, so the GLM
# above (which needs numeric 0/1 clm) is unaffected.

train_data_rf <- train_data
test_data_rf  <- test_data

train_data_rf$clm <- factor(train_data_rf$clm, levels = c(0, 1))
test_data_rf$clm  <- factor(test_data_rf$clm,  levels = c(0, 1))

set.seed(123)

rf_occurrence <- randomForest(
  occurrence_formula,
  data = train_data_rf,
  ntree = 500,
  importance = TRUE
)

rf_occurrence_probability <- predict(
  rf_occurrence,
  newdata = test_data_rf,
  type = "prob"
)[, "1"]

# 5.2 Claim Severity Random Forest

set.seed(123)

rf_severity <- randomForest(
  severity_formula,
  data = severity_train,
  ntree = 500,
  importance = TRUE
)

rf_severity_prediction <- predict(rf_severity, newdata = test_data)

rf_expected_cost <- rf_occurrence_probability * rf_severity_prediction

# ------------------------------------------------------------
# 6. Predictive Performance
# ------------------------------------------------------------

actual_cost <- test_data$claimcst0

performance <- data.frame(
  Model = c("Two-Part GLM", "Two-Part Random Forest"),
  RMSE = c(
    RMSE(glm_expected_cost, actual_cost),
    RMSE(rf_expected_cost, actual_cost)
  ),
  MAE = c(
    MAE(glm_expected_cost, actual_cost),
    MAE(rf_expected_cost, actual_cost)
  ),
  Rsquared = c(
    R2(glm_expected_cost, actual_cost),
    R2(rf_expected_cost, actual_cost)
  )
)

print(performance)

write.csv(
  performance,
  "tables/B02_model_performance.csv",
  row.names = FALSE
)

# ------------------------------------------------------------
# 7. Model Comparison Plot
# ------------------------------------------------------------

performance_long <- performance %>%
  pivot_longer(
    cols = c(RMSE, MAE, Rsquared),
    names_to = "Metric",
    values_to = "Value"
  )

p <- ggplot(
  performance_long,
  aes(x = Metric, y = Value, fill = Model)
) +
  geom_col(position = "dodge") +
  labs(
    title = "Two-Part GLM vs Two-Part Random Forest",
    x = "Performance Metric",
    y = "Value"
  ) +
  theme_minimal()

ggsave(
  "figures/B02_model_comparison.png",
  p,
  width = 8, height = 5, dpi = 300
)

# ------------------------------------------------------------
# 8. Save Models
# ------------------------------------------------------------

saveRDS(occurrence_glm, "results/B02_occurrence_glm.rds")
saveRDS(severity_glm, "results/B02_severity_glm.rds")
saveRDS(rf_occurrence, "results/B02_rf_occurrence.rds")
saveRDS(rf_severity, "results/B02_rf_severity.rds")

# ------------------------------------------------------------
# 9. Save Predictions
# ------------------------------------------------------------

prediction_results <- test_data %>%
  select(clm, claimcst0) %>%
  mutate(
    GLM_Predicted = glm_expected_cost,
    RF_Predicted = rf_expected_cost
  )

write.csv(
  prediction_results,
  "tables/B02_test_predictions.csv",
  row.names = FALSE
)

cat("B02 completed successfully.\n")