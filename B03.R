# ============================================================
# B03: Insurance SHAP Explanations
# Random Forest Claim Severity
# ============================================================

library(tidyverse)
library(caret)
library(randomForest)
library(kernelshap)
library(ggplot2)

# ------------------------------------------------------------
# 1. Load Data
# ------------------------------------------------------------

insurance_data <- readRDS("results/insurance_data.rds")
severity_data  <- readRDS("results/severity_data.rds")
rf_severity    <- readRDS("results/B02_rf_severity.rds")

# ------------------------------------------------------------
# 2. Recreate Train-Test Split
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

# ------------------------------------------------------------
# 3. Predictor Set
# ------------------------------------------------------------
# Must match exactly what rf_severity was trained on in B02.R:
# claimcst0 (response), clm (constant within this subset), and
# numclaims (excluded as leakage) are all dropped.

severity_predictors <- setdiff(
  names(severity_train),
  c("claimcst0", "clm", "numclaims")
)

# ------------------------------------------------------------
# 4. SHAP Background Sample
# ------------------------------------------------------------

set.seed(123)

bg_sample <- severity_train[
  sample(nrow(severity_train), min(100, nrow(severity_train))),
  severity_predictors,
  drop = FALSE
]

# ------------------------------------------------------------
# 5. SHAP Explanation Sample
# ------------------------------------------------------------

set.seed(123)

explain_sample <- severity_test[
  sample(nrow(severity_test), min(100, nrow(severity_test))),
  severity_predictors,
  drop = FALSE
]

# ------------------------------------------------------------
# 6. Local SHAP
# ------------------------------------------------------------

new_observation <- explain_sample[1, , drop = FALSE]

local_shap <- kernelshap(
  object = rf_severity,
  X = new_observation,
  bg_X = bg_sample
)

local_results <- data.frame(
  Feature = colnames(local_shap$S),
  SHAP = as.numeric(local_shap$S[1, ])
) %>%
  arrange(desc(abs(SHAP)))

print(local_results)

write.csv(
  local_results,
  "tables/B03_local_shap.csv",
  row.names = FALSE
)

# ------------------------------------------------------------
# 7. Local SHAP Plot
# ------------------------------------------------------------

local_plot <- ggplot(
  local_results,
  aes(x = reorder(Feature, abs(SHAP)), y = SHAP)
) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Local SHAP Explanation of Claim Severity",
    x = "Feature",
    y = "SHAP Value"
  ) +
  theme_minimal()

ggsave(
  "figures/B03_local_shap.png",
  local_plot,
  width = 7, height = 5, dpi = 300
)

# ------------------------------------------------------------
# 8. Global SHAP
# ------------------------------------------------------------

global_shap_values <- kernelshap(
  object = rf_severity,
  X = explain_sample,
  bg_X = bg_sample
)

saveRDS(global_shap_values, "results/B03_global_shap_values.rds")

global_shap <- data.frame(
  Feature = colnames(global_shap_values$S),
  Mean_Absolute_SHAP = colMeans(abs(global_shap_values$S))
) %>%
  arrange(desc(Mean_Absolute_SHAP))

print(global_shap)

write.csv(
  global_shap,
  "tables/B03_global_shap.csv",
  row.names = FALSE
)

# ------------------------------------------------------------
# 9. Global SHAP Plot
# ------------------------------------------------------------

global_plot <- ggplot(
  global_shap,
  aes(x = reorder(Feature, Mean_Absolute_SHAP), y = Mean_Absolute_SHAP)
) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Global SHAP Importance for Claim Severity",
    x = "Feature",
    y = "Mean Absolute SHAP"
  ) +
  theme_minimal()

ggsave(
  "figures/B03_global_shap.png",
  global_plot,
  width = 7, height = 5, dpi = 300
)

# ------------------------------------------------------------
# 10. SHAP Efficiency Check
# ------------------------------------------------------------

local_prediction <- predict(rf_severity, newdata = new_observation)

baseline_prediction <- mean(predict(rf_severity, newdata = bg_sample))

shap_sum <- sum(local_shap$S[1, ])

efficiency_result <- data.frame(
  Model_Prediction = local_prediction,
  Baseline = baseline_prediction,
  SHAP_Sum = shap_sum,
  Baseline_Plus_SHAP = baseline_prediction + shap_sum,
  Difference = local_prediction - (baseline_prediction + shap_sum)
)

stopifnot(
  "SHAP efficiency property violated" =
    abs(efficiency_result$Difference) < 1e-6
)

print(efficiency_result)

write.csv(
  efficiency_result,
  "tables/B03_efficiency_check.csv",
  row.names = FALSE
)

cat("B03 completed successfully.\n")