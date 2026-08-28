# ============================================================
# B06: Insurance SHAP Stability
# Under Model Overspecification (Severity Random Forest)
# Mirrors A07.R. Scaled down from A07's 100 reps against mtcars'
# full test set to 5 reps against a 50-row background/explain
# sample (matching B04/B05's scale), for the same computational-
# feasibility reasons. State this explicitly in 4.7/4.8.
#
# TIME CHECK: test with runs <- 2 first and time it before committing
# to the full run for both Base and Overspecified models.
# ============================================================

library(tidyverse)
library(caret)
library(randomForest)
library(kernelshap)

# ------------------------------------------------------------
# 1. Load Data
# ------------------------------------------------------------

insurance_data <- readRDS("results/insurance_data.rds")

# ------------------------------------------------------------
# 2. Recreate Train-Test Split (matches B02/B03/B04/B05)
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
# 3. Predictor Set (same exclusions as B02/B03/B04/B05)
# ------------------------------------------------------------

predictors_base <- setdiff(
  names(severity_train),
  c("claimcst0", "clm", "numclaims")
)

model_formula_base <- as.formula(
  paste("claimcst0 ~", paste(predictors_base, collapse = " + "))
)

# ------------------------------------------------------------
# 4. Overspecified Version
# Same severity data with 5 independent Gaussian noise predictors
# added (same construction as A07).
# ------------------------------------------------------------

set.seed(42)

severity_train_over <- severity_train
severity_test_over  <- severity_test

for (i in 1:5) {
  severity_train_over[[paste0("noise_", i)]] <- rnorm(nrow(severity_train_over))
  severity_test_over[[paste0("noise_", i)]]  <- rnorm(nrow(severity_test_over))
}

predictors_over <- c(predictors_base, paste0("noise_", 1:5))

model_formula_over <- as.formula(
  paste("claimcst0 ~", paste(predictors_over, collapse = " + "))
)

# ------------------------------------------------------------
# 5. Fixed Sample Row Indices
# Shared between base and overspecified so the same policies are
# explained in both conditions.
# ------------------------------------------------------------

set.seed(123)
sample_idx_bg <- sample(nrow(severity_train), min(50, nrow(severity_train)))
set.seed(123)
sample_idx_explain <- sample(nrow(severity_test), min(50, nrow(severity_test)))

bg_base      <- severity_train[sample_idx_bg, predictors_base, drop = FALSE]
explain_base <- severity_test[sample_idx_explain, predictors_base, drop = FALSE]

bg_over      <- severity_train_over[sample_idx_bg, predictors_over, drop = FALSE]
explain_over <- severity_test_over[sample_idx_explain, predictors_over, drop = FALSE]

runs <- 5   # scaled down from A07's 100 - see note above

# ------------------------------------------------------------
# 6. Base Random Forest Stability
# ------------------------------------------------------------

base_results <- data.frame()

for (seed in 1:runs) {
  
  cat("Base run", seed, "starting\n")
  
  set.seed(seed)
  
  rf <- randomForest(
    model_formula_base,
    data = severity_train,
    ntree = 500
  )
  
  shap <- kernelshap(object = rf, X = explain_base, bg_X = bg_base)
  
  base_results <- bind_rows(
    base_results,
    tibble(
      Model = "Base",
      Run = seed,
      Feature = colnames(shap$S),
      Mean_Abs_SHAP = colMeans(abs(shap$S))
    )
  )
  
  cat("Base run", seed, "complete\n")
}

# ------------------------------------------------------------
# 7. Overspecified Random Forest Stability
# ------------------------------------------------------------

over_results <- data.frame()

for (seed in 1:runs) {
  
  cat("Overspecified run", seed, "starting\n")
  
  set.seed(seed)
  
  rf <- randomForest(
    model_formula_over,
    data = severity_train_over,
    ntree = 500
  )
  
  shap <- kernelshap(object = rf, X = explain_over, bg_X = bg_over)
  
  over_results <- bind_rows(
    over_results,
    tibble(
      Model = "Overspecified",
      Run = seed,
      Feature = colnames(shap$S),
      Mean_Abs_SHAP = colMeans(abs(shap$S))
    )
  )
  
  cat("Overspecified run", seed, "complete\n")
}

# ------------------------------------------------------------
# 8. Combine and Summarise
# ------------------------------------------------------------

all_results <- bind_rows(base_results, over_results)

stability_summary <- all_results %>%
  group_by(Model, Feature) %>%
  summarise(
    Mean_Absolute_SHAP = mean(Mean_Abs_SHAP),
    SD_SHAP = sd(Mean_Abs_SHAP),
    .groups = "drop"
  ) %>%
  arrange(Model, desc(Mean_Absolute_SHAP))

write.csv(
  stability_summary,
  "tables/B06_overspecified_stability.csv",
  row.names = FALSE
)

sd_summary <- all_results %>%
  group_by(Model, Feature) %>%
  summarise(SD_SHAP = sd(Mean_Abs_SHAP), .groups = "drop")

p <- ggplot(
  sd_summary,
  aes(x = reorder(Feature, SD_SHAP), y = SD_SHAP, fill = Model)
) +
  geom_col(position = "dodge") +
  coord_flip() +
  scale_fill_manual(
    values = c("Base" = "steelblue", "Overspecified" = "tomato")
  ) +
  labs(
    title = "Comparison of SHAP Stability Between Base and Overspecified Severity Models (Insurance)",
    x = "Feature",
    y = "Standard Deviation of Mean Absolute SHAP Values"
  ) +
  theme_minimal()

ggsave(
  "figures/B06_overspecified_stability.png",
  plot = p,
  width = 9, height = 6, dpi = 300
)

# ------------------------------------------------------------
# Mean Absolute SHAP comparison (Base vs Overspecified)
# ------------------------------------------------------------
# The SD plot above shows STABILITY, not IMPORTANCE - it does not show
# whether noise variables actually receive comparable importance to
# genuine predictors. This plot does: feature order is fixed by Base
# importance (descending), with noise variables grouped after a dashed
# separator, so real predictors and noise variables are compared on the
# same scale directly.

base_order <- stability_summary %>%
  filter(Model == "Base") %>%
  arrange(desc(Mean_Absolute_SHAP)) %>%
  pull(Feature)

noise_order <- sort(unique(stability_summary$Feature[
  grepl("^noise_", stability_summary$Feature)
]))

feature_order <- c(base_order, noise_order)

importance_plot <- ggplot(
  stability_summary,
  aes(
    x = factor(Feature, levels = rev(feature_order)),
    y = Mean_Absolute_SHAP,
    fill = Model
  )
) +
  geom_col(position = "dodge") +
  coord_flip() +
  geom_vline(
    xintercept = length(noise_order) + 0.5,
    linetype = "dashed", color = "grey50"
  ) +
  scale_fill_manual(
    values = c("Base" = "steelblue", "Overspecified" = "tomato")
  ) +
  labs(
    title = "Mean Absolute SHAP: Base vs Overspecified Severity Models (Insurance)",
    x = "Feature",
    y = "Mean Absolute SHAP"
  ) +
  theme_minimal()

ggsave(
  "figures/B06_mean_importance_comparison.png",
  plot = importance_plot,
  width = 9, height = 6, dpi = 300
)

print(stability_summary, n = Inf)
cat("\nB06 completed successfully.\n")