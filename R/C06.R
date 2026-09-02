# C06: SurvSHAP(t) Stability Under Model Overspecification
# Random Survival Forest

library(tidyverse)
library(survival)
library(ranger)
library(survex)

survival_train <- readRDS("results/survival_train.rds")
survival_test  <- readRDS("results/survival_test.rds")
survival_predictors <- readRDS("results/survival_predictors.rds")

model_formula_base <- as.formula(
  paste("Surv(time, cens) ~", paste(survival_predictors, collapse = " + "))
)

set.seed(42)

survival_train_over <- survival_train
survival_test_over  <- survival_test

for (i in 1:5) {
  survival_train_over[[paste0("noise_", i)]] <- rnorm(nrow(survival_train_over))
  survival_test_over[[paste0("noise_", i)]]  <- rnorm(nrow(survival_test_over))
}

predictors_over <- c(survival_predictors, paste0("noise_", 1:5))

model_formula_over <- as.formula(
  paste("Surv(time, cens) ~", paste(predictors_over, collapse = " + "))
)

set.seed(123)
explain_idx <- sample(nrow(survival_test), min(15, nrow(survival_test)))

explain_base <- survival_test[explain_idx, survival_predictors]
explain_over <- survival_test_over[explain_idx, predictors_over]

runs <- 5

run_condition <- function(formula, train_data, test_data_full,
                          predictors, explain_data, seed, label) {
  
  set.seed(seed)
  
  rsf_temp <- ranger(formula, data = train_data, num.trees = 500, seed = seed)
  
  temp_explainer <- explain(
    rsf_temp,
    data = test_data_full[, predictors],
    y = Surv(test_data_full$time, test_data_full$cens),
    label = label,
    verbose = FALSE
  )
  
  obs_shap_list <- list()
  for (i in seq_len(nrow(explain_data))) {
    obs_shap <- predict_parts(
      temp_explainer,
      new_observation = explain_data[i, , drop = FALSE],
      type = "survshap"
    )
    obs_shap_list[[i]] <- as.data.frame(obs_shap$result)[, predictors]
  }
  
  obs_importance <- map_dfr(obs_shap_list, ~ colMeans(abs(.x), na.rm = TRUE))
  
  obs_importance %>%
    summarise(across(everything(), ~ mean(.x, na.rm = TRUE))) %>%
    pivot_longer(everything(), names_to = "Feature", values_to = "Mean_Abs_SHAP")
}

base_results <- data.frame()

for (run in 1:runs) {
  cat("Base run", run, "starting\n")
  run_start <- Sys.time()
  
  result <- run_condition(
    model_formula_base, survival_train, survival_test,
    survival_predictors, explain_base, run, paste0("RSF base run ", run)
  )
  result$Model <- "Base"
  result$Run <- run
  
  base_results <- bind_rows(base_results, result)
  
  cat("Base run", run, "complete in",
      round(as.numeric(Sys.time() - run_start, units = "mins"), 2), "min\n")
}

over_results <- data.frame()

for (run in 1:runs) {
  cat("Overspecified run", run, "starting\n")
  run_start <- Sys.time()
  
  result <- run_condition(
    model_formula_over, survival_train_over, survival_test_over,
    predictors_over, explain_over, run, paste0("RSF overspecified run ", run)
  )
  result$Model <- "Overspecified"
  result$Run <- run
  
  over_results <- bind_rows(over_results, result)
  
  cat("Overspecified run", run, "complete in",
      round(as.numeric(Sys.time() - run_start, units = "mins"), 2), "min\n")
}

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
  "tables/C06_overspecified_stability.csv",
  row.names = FALSE
)

sd_plot <- ggplot(
  stability_summary,
  aes(x = reorder(Feature, SD_SHAP), y = SD_SHAP, fill = Model)
) +
  geom_col(position = "dodge") +
  coord_flip() +
  scale_fill_manual(values = c("Base" = "steelblue", "Overspecified" = "tomato")) +
  labs(
    title = "SurvSHAP(t) Stability: Base vs Overspecified (Survival RSF)",
    x = "Feature", y = "Standard Deviation of Mean Absolute SHAP"
  ) +
  theme_minimal()

ggsave("figures/C06_overspecified_stability.png", plot = sd_plot,
       width = 9, height = 6, dpi = 300)

base_order <- stability_summary %>%
  filter(Model == "Base") %>%
  arrange(desc(Mean_Absolute_SHAP)) %>%
  pull(Feature)
noise_order <- sort(unique(stability_summary$Feature[grepl("^noise_", stability_summary$Feature)]))
feature_order <- c(base_order, noise_order)

importance_plot <- ggplot(
  stability_summary,
  aes(x = factor(Feature, levels = rev(feature_order)), y = Mean_Absolute_SHAP, fill = Model)
) +
  geom_col(position = "dodge") +
  coord_flip() +
  geom_vline(xintercept = length(noise_order) + 0.5, linetype = "dashed", color = "grey50") +
  scale_fill_manual(values = c("Base" = "steelblue", "Overspecified" = "tomato")) +
  labs(
    title = "Mean Absolute SurvSHAP(t): Base vs Overspecified (Survival RSF)",
    x = "Feature", y = "Mean Absolute SHAP"
  ) +
  theme_minimal()

ggsave("figures/C06_mean_importance_comparison.png", plot = importance_plot,
       width = 9, height = 6, dpi = 300)

print(stability_summary, n = Inf)