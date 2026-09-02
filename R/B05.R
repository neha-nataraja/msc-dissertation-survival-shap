# ============================================================
# B05: Insurance SHAP Stability
# Across Different Numbers of Trees (Severity Random Forest)
# Mirrors A06.R. Scaled down from A06's 100 reps against mtcars'
# 6-row test set: insurance uses a 50-row background/explain sample
# with 5 reps per tree count (matching B04's 5-seed scale), since
# KernelSHAP cost scales with both sample size and repetitions.
# State this explicitly as a deliberate computational choice in 4.7.
#
# TIME CHECK: test with tree_values <- c(50) and runs <- 2 first and
# time it before committing to the full sweep.
# ============================================================

library(tidyverse)
library(caret)
library(randomForest)
library(kernelshap)

dir.create("tables", showWarnings = FALSE)
dir.create("figures", showWarnings = FALSE)
dir.create("results", showWarnings = FALSE)

# ------------------------------------------------------------
# 1. Load Data
# ------------------------------------------------------------

insurance_data <- readRDS("results/insurance_data.rds")

# ------------------------------------------------------------
# 2. Recreate Train-Test Split (matches B02/B03/B04)
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
# 3. Predictor Set (same exclusions as B02/B03/B04)
# ------------------------------------------------------------

severity_predictors <- setdiff(
  names(severity_train),
  c("claimcst0", "clm", "numclaims")
)

severity_formula <- as.formula(
  paste("claimcst0 ~", paste(severity_predictors, collapse = " + "))
)

# ------------------------------------------------------------
# 4. Fixed SHAP Samples
# ------------------------------------------------------------

set.seed(123)

bg_sample <- severity_train[
  sample(nrow(severity_train), min(50, nrow(severity_train))),
  severity_predictors,
  drop = FALSE
]

set.seed(123)

explain_sample <- severity_test[
  sample(nrow(severity_test), min(50, nrow(severity_test))),
  severity_predictors,
  drop = FALSE
]

# ------------------------------------------------------------
# 5. Tree-Count Sweep
# ------------------------------------------------------------

tree_values <- c(50, 100, 250, 500, 1000)
runs <- 5   # scaled down from A06's 100 - see note above

stability_results <- data.frame()

for (ntree in tree_values) {
  
  for (seed in 1:runs) {
    
    cat("Starting ntree =", ntree, " run =", seed, "\n")
    
    set.seed(seed)
    
    rf_temp <- randomForest(
      severity_formula,
      data = severity_train,
      ntree = ntree
    )
    
    shap_temp <- kernelshap(
      object = rf_temp,
      X = explain_sample,
      bg_X = bg_sample
    )
    
    temp_results <- tibble(
      Trees = ntree,
      Run = seed,
      Feature = colnames(shap_temp$S),
      Mean_Absolute_SHAP = colMeans(abs(shap_temp$S))
    )
    
    stability_results <- bind_rows(stability_results, temp_results)
    
    cat("Completed ntree =", ntree, " run =", seed, "\n")
  }
}

# ------------------------------------------------------------
# 6. Summary Table
# ------------------------------------------------------------

stability_summary <- stability_results %>%
  group_by(Trees, Feature) %>%
  summarise(
    SD_SHAP = sd(Mean_Absolute_SHAP),
    Mean_Absolute_SHAP = mean(Mean_Absolute_SHAP),
    .groups = "drop"
  )

write.csv(
  stability_summary,
  "tables/B05_shap_tree_stability.csv",
  row.names = FALSE
)

saveRDS(stability_results, "results/B05_shap_tree_stability.rds")

# ------------------------------------------------------------
# 7. Plots
# ------------------------------------------------------------

ggplot(
  stability_results,
  aes(x = factor(Trees), y = Mean_Absolute_SHAP)
) +
  geom_boxplot() +
  facet_wrap(~Feature, scales = "free_y") +
  labs(
    title = "SHAP Stability Across Different Numbers of Trees (Insurance Severity)",
    x = "Number of Trees",
    y = "Mean Absolute SHAP"
  ) +
  theme_minimal()

ggsave("figures/B05_tree_stability_boxplot.png", width = 12, height = 8)

sd_plot <- stability_summary %>%
  group_by(Trees) %>%
  summarise(Average_SD = mean(SD_SHAP), .groups = "drop")

ggplot(sd_plot, aes(x = Trees, y = Average_SD)) +
  geom_line() +
  geom_point(size = 2) +
  labs(
    title = "Average SHAP Stability Across Tree Numbers (Insurance Severity)",
    x = "Number of Trees",
    y = "Average Standard Deviation"
  ) +
  theme_minimal()

ggsave("figures/B05_tree_stability_sd.png", width = 8, height = 6)

print(stability_summary, n = 50)
cat("B05 completed successfully.\n")