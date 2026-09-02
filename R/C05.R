# C05: SurvSHAP(t) Stability Across Different Numbers of Trees

library(tidyverse)
library(survival)
library(ranger)
library(survex)

survival_train <- readRDS("results/survival_train.rds")
survival_test  <- readRDS("results/survival_test.rds")
survival_predictors <- readRDS("results/survival_predictors.rds")
survival_formula <- readRDS("results/survival_formula.rds")

severity_test <- survival_test
set.seed(123)

explain_idx <- sample(nrow(survival_test), min(15, nrow(survival_test)))
explain_sample <- survival_test[explain_idx, survival_predictors]

tree_values <- c(50, 100, 250, 500, 1000)
runs <- 3   
stability_results <- data.frame()

for (ntree in tree_values) {
  
  for (run in 1:runs) {
    
    cat("Starting ntree =", ntree, " run =", run, "\n")
    run_start <- Sys.time()
    
    set.seed(run)
    
    rsf_temp <- ranger(
      survival_formula,
      data = survival_train,
      num.trees = ntree,
      seed = run
    )
    
    temp_explainer <- explain(
      rsf_temp,
      data = survival_test[, survival_predictors],
      y = Surv(survival_test$time, survival_test$cens),
      label = paste0("RSF ntree=", ntree, " run=", run),
      verbose = FALSE
    )
    
    obs_shap_list <- list()
    
    for (i in seq_len(nrow(explain_sample))) {
      obs_shap <- predict_parts(
        temp_explainer,
        new_observation = explain_sample[i, , drop = FALSE],
        type = "survshap"
      )
      obs_shap_list[[i]] <- as.data.frame(obs_shap$result)[, survival_predictors]
    }
    
    obs_importance <- map_dfr(obs_shap_list, ~ colMeans(abs(.x), na.rm = TRUE))
    
    run_summary <- obs_importance %>%
      summarise(across(everything(), ~ mean(.x, na.rm = TRUE))) %>%
      pivot_longer(everything(), names_to = "Feature", values_to = "Mean_Absolute_SHAP") %>%
      mutate(Trees = ntree, Run = run)
    
    stability_results <- bind_rows(stability_results, run_summary)
    
    cat("Completed ntree =", ntree, " run =", run, "in",
        round(as.numeric(Sys.time() - run_start, units = "mins"), 2), "min\n")
    
    rm(rsf_temp, temp_explainer, obs_shap_list, obs_importance, run_summary)
    gc(verbose = FALSE)
  }

  write.csv(
    stability_results,
    "tables/C05_shap_tree_stability_partial.csv",
    row.names = FALSE
  )
  cat("Saved partial results through ntree =", ntree, "\n")
}

stability_summary <- stability_results %>%
  group_by(Trees, Feature) %>%
  summarise(
    SD_SHAP = sd(Mean_Absolute_SHAP),
    Mean_Absolute_SHAP = mean(Mean_Absolute_SHAP),
    .groups = "drop"
  )

write.csv(
  stability_summary,
  "tables/C05_shap_tree_stability.csv",
  row.names = FALSE
)

saveRDS(stability_results, "results/C05_shap_tree_stability.rds")

ggplot(
  stability_results,
  aes(x = factor(Trees), y = Mean_Absolute_SHAP)
) +
  geom_boxplot() +
  facet_wrap(~Feature, scales = "free_y") +
  labs(
    title = "SurvSHAP(t) Stability Across Different Numbers of Trees (Survival RSF)",
    x = "Number of Trees",
    y = "Mean Absolute SHAP"
  ) +
  theme_minimal()

ggsave("figures/C05_tree_stability_boxplot.png", width = 12, height = 8)

sd_plot <- stability_summary %>%
  group_by(Trees) %>%
  summarise(Average_SD = mean(SD_SHAP), .groups = "drop")

ggplot(sd_plot, aes(x = Trees, y = Average_SD)) +
  geom_line() +
  geom_point(size = 2) +
  labs(
    title = "Average SurvSHAP(t) Stability Across Tree Numbers (Survival RSF)",
    x = "Number of Trees",
    y = "Average Standard Deviation"
  ) +
  theme_minimal()

ggsave("figures/C05_tree_stability_sd.png", width = 8, height = 6)

print(stability_summary, n = 50)