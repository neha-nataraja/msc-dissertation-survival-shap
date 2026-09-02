# ============================================================
# B04: Insurance SHAP Stability
# Random Seed Stability
# ============================================================

library(tidyverse)
library(caret)
library(randomForest)
library(kernelshap)

# ------------------------------------------------------------
# 1. Load Data
# ------------------------------------------------------------

insurance_data <- readRDS("results/insurance_data.rds")
severity_data  <- readRDS("results/severity_data.rds")

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
# Same exclusions as B02/B03: claimcst0 (response), clm (constant
# within this subset), numclaims (leakage).

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

stability_explain <- severity_test[
  sample(nrow(severity_test), min(50, nrow(severity_test))),
  severity_predictors,
  drop = FALSE
]

set.seed(123)

stability_background <- severity_train[
  sample(nrow(severity_train), min(50, nrow(severity_train))),
  severity_predictors,
  drop = FALSE
]

# ------------------------------------------------------------
# 5. Seeds
# ------------------------------------------------------------

seeds <- c(
  123, 456, 789, 1011, 2022,
  3033, 4044, 5055, 6066, 7077,
  8088, 9099, 1111, 2222, 3333,
  4444, 5555, 6666, 7777, 8888
)

# Currently set to the first 5 (already timed and confirmed to run).
# If B02/B03's runtime gives you enough headroom today, extend this
# to more seeds - e.g. test_seeds <- seeds[1:10] - and re-run. Time
# it again before jumping straight to all 20.

test_seeds <- seeds[1:5]

# ------------------------------------------------------------
# 6. Stability Loop
# ------------------------------------------------------------

stability_results <- data.frame()

for (s in test_seeds) {
  
  cat("Starting seed:", s, "\n")
  
  set.seed(s)
  
  rf_temp <- randomForest(
    severity_formula,
    data = severity_train,
    ntree = 500,
    importance = TRUE
  )
  
  shap_temp <- kernelshap(
    object = rf_temp,
    X = stability_explain,
    bg_X = stability_background
  )
  
  temp_results <- data.frame(
    Seed = s,
    Feature = colnames(shap_temp$S),
    Mean_Absolute_SHAP = colMeans(abs(shap_temp$S)),
    SD_SHAP = apply(shap_temp$S, 2, sd)
  )
  
  stability_results <- bind_rows(stability_results, temp_results)
  
  cat("Completed seed:", s, "\n")
}

# ------------------------------------------------------------
# 7. Save Raw Per-Seed Results
# ------------------------------------------------------------

write.csv(
  stability_results,
  "tables/B04_seed_stability_raw.csv",
  row.names = FALSE
)

# ------------------------------------------------------------
# 8. Summary Across Seeds
# ------------------------------------------------------------
# This is the table that matches Chapter 3's stability tables (e.g.
# Table 6): for each feature, its Mean Absolute SHAP averaged across
# seeds and the standard deviation of that value ACROSS seeds - not
# to be confused with SD_SHAP above, which is the within-sample SHAP
# spread for a single seed.

stability_summary <- stability_results %>%
  group_by(Feature) %>%
  summarise(
    Mean_SHAP = mean(Mean_Absolute_SHAP),
    SD_Across_Seeds = sd(Mean_Absolute_SHAP),
    .groups = "drop"
  ) %>%
  arrange(desc(Mean_SHAP))

print(stability_summary)

write.csv(
  stability_summary,
  "tables/B04_seed_stability_summary.csv",
  row.names = FALSE
)

# ------------------------------------------------------------
# 9. Stability Plot
# ------------------------------------------------------------

stability_plot <- ggplot(
  stability_summary,
  aes(x = reorder(Feature, Mean_SHAP), y = Mean_SHAP)
) +
  geom_col(fill = "steelblue") +
  geom_errorbar(
    aes(ymin = Mean_SHAP - SD_Across_Seeds, ymax = Mean_SHAP + SD_Across_Seeds),
    width = 0.2
  ) +
  coord_flip() +
  labs(
    title = paste0(
      "SHAP Stability Across ", length(test_seeds),
      " Random Seeds (Insurance Severity Model)"
    ),
    x = "Feature",
    y = "Mean Absolute SHAP"
  ) +
  theme_minimal()

ggsave(
  "figures/B04_seed_stability.png",
  stability_plot,
  width = 7, height = 5, dpi = 300
)

cat("B04 completed successfully (", length(test_seeds), "seeds ).\n")