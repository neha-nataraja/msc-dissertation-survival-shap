# C04: SurvSHAP(t) Stability Across Random Seeds

library(tidyverse)
library(survival)
library(ranger)
library(survex)

survival_train <- readRDS("results/survival_train.rds")
survival_test  <- readRDS("results/survival_test.rds")
survival_predictors <- readRDS("results/survival_predictors.rds")
survival_formula <- readRDS("results/survival_formula.rds")

set.seed(123)

stability_sample_idx <- sample(nrow(survival_test), min(3, nrow(survival_test)))
stability_sample <- survival_test[stability_sample_idx, survival_predictors]

seeds <- c(123, 456, 789, 1011, 2022)

stability_results <- data.frame()

for (s in seeds) {
  
  cat("Starting seed:", s, "\n")
  seed_start <- Sys.time()
  
  set.seed(s)
  
  rsf_temp <- ranger(
    survival_formula,
    data = survival_train,
    num.trees = 500,
    seed = s
  )
  
  temp_explainer <- explain(
    rsf_temp,
    data = survival_test[, survival_predictors],
    y = Surv(survival_test$time, survival_test$cens),
    label = paste0("RSF seed ", s),
    verbose = FALSE
  )
  
  cat("  Explaining", nrow(stability_sample), "observations for this seed...\n")
  
  seed_shap_list <- list()
  
  for (i in seq_len(nrow(stability_sample))) {
    obs_shap <- predict_parts(
      temp_explainer,
      new_observation = stability_sample[i, , drop = FALSE],
      type = "survshap"
    )
    seed_shap_list[[i]] <- as.data.frame(obs_shap$result)[, survival_predictors]
  }
  
  temp_importance <- map_dfr(seed_shap_list, ~ colMeans(abs(.x), na.rm = TRUE)) %>%
    summarise(across(everything(), mean, na.rm = TRUE)) %>%
    pivot_longer(everything(), names_to = "Feature", values_to = "Mean_Absolute_SHAP") %>%
    mutate(Seed = s)
  
  stability_results <- bind_rows(stability_results, temp_importance)
  
  cat("Completed seed:", s, "in",
      round(as.numeric(Sys.time() - seed_start, units = "mins"), 2), "min\n")
}

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
  "tables/C04_seed_stability_summary.csv",
  row.names = FALSE
)

write.csv(
  stability_results,
  "tables/C04_seed_stability_raw.csv",
  row.names = FALSE
)
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
    title = "SurvSHAP(t) Stability Across 5 Random Seeds (Survival RSF)",
    x = "Feature",
    y = "Mean Absolute SHAP (averaged over time and observations)"
  ) +
  theme_minimal()

ggsave(
  "figures/C04_seed_stability.png",
  stability_plot,
  width = 7, height = 5, dpi = 300
)