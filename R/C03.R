# C03: SurvSHAP(t) Explanations
# Random Survival Forest (the black-box model)

library(tidyverse)
library(survival)
library(ranger)
library(survex)

survival_test <- readRDS("results/survival_test.rds")
survival_predictors <- readRDS("results/survival_predictors.rds")
rsf_explainer <- readRDS("results/C02_rsf_explainer.rds")

new_observation <- survival_test[1, survival_predictors]

cat("Computing local SurvSHAP(t)...\n")

local_start <- Sys.time()

local_survshap <- predict_parts(
  rsf_explainer,
  new_observation = new_observation,
  type = "survshap"
)

cat("Local SurvSHAP(t) computed in:",
    round(as.numeric(Sys.time() - local_start), 2), "sec\n")

print(local_survshap)

local_df <- as.data.frame(local_survshap$result)

local_df$Time <- as.numeric(gsub("^t=", "", rownames(local_survshap$result)))
local_df <- local_df[, c("Time", setdiff(names(local_df), "Time"))]

write.csv(
  local_df,
  "tables/C03_local_survshap.csv",
  row.names = FALSE
)

local_plot <- plot(local_survshap)

ggsave(
  "figures/C03_local_survshap.png",
  local_plot,
  width = 8, height = 5, dpi = 300
)
# Global SurvSHAP(t)

set.seed(123)

global_sample_idx <- sample(nrow(survival_test), min(5, nrow(survival_test)))
global_sample <- survival_test[global_sample_idx, survival_predictors]

cat("Computing global SurvSHAP(t) over", nrow(global_sample), "observations\n")
cat("(looping predict_parts() individually, known ~52 sec/obs from the\n")
cat("local step above - model_survshap()'s built-in aggregation was tried\n")
cat("and found far more expensive, so it is not used here)\n")

global_start <- Sys.time()

global_shap_list <- list()

for (i in seq_len(nrow(global_sample))) {
  
  obs_start <- Sys.time()
  cat("  Observation", i, "of", nrow(global_sample), "...\n")
  
  obs_shap <- predict_parts(
    rsf_explainer,
    new_observation = global_sample[i, , drop = FALSE],
    type = "survshap"
  )
  
  global_shap_list[[i]] <- as.data.frame(obs_shap$result)[, survival_predictors]
  
  cat("  done in", round(as.numeric(Sys.time() - obs_start), 1), "sec\n")
}
.

per_observation_importance <- map_dfr(
  global_shap_list,
  ~ colMeans(abs(.x), na.rm = TRUE)
)

global_importance <- per_observation_importance %>%
  summarise(across(everything(), mean, na.rm = TRUE)) %>%
  pivot_longer(everything(), names_to = "Feature", values_to = "Mean_Absolute_SHAP") %>%
  arrange(desc(Mean_Absolute_SHAP))

print(global_importance)

write.csv(
  global_importance,
  "tables/C03_global_survshap_importance.csv",
  row.names = FALSE
)

global_plot <- ggplot(
  global_importance,
  aes(x = reorder(Feature, Mean_Absolute_SHAP), y = Mean_Absolute_SHAP)
) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(
    title = "Global SurvSHAP(t) Feature Importance (Survival RSF)",
    x = "Feature",
    y = "Mean Absolute SHAP (averaged over time and observations)"
  ) +
  theme_minimal()

ggsave(
  "figures/C03_global_survshap.png",
  global_plot,
  width = 8, height = 5, dpi = 300
)

saveRDS(global_shap_list, "results/C03_global_survshap_list.rds")