library(tidyverse)
library(survival)
library(ranger)
library(survex)

survival_train <- readRDS("results/survival_train.rds")
survival_test  <- readRDS("results/survival_test.rds")
survival_predictors <- readRDS("results/survival_predictors.rds")
survival_formula <- readRDS("results/survival_formula.rds")

# Cox Proportional Hazards Model
cox_model <- coxph(
  survival_formula,
  data = survival_train,
  x = TRUE
)

cat("Cox model fitted.\n")

# Random Survival Forest (ranger)
set.seed(123)

rsf_model <- ranger(
  survival_formula,
  data = survival_train,
  num.trees = 500,
  seed = 123
)

cat("Random Survival Forest fitted.\n")

# Build survex Explainers
cox_explainer <- explain(
  cox_model,
  data = survival_test[, survival_predictors],
  y = Surv(survival_test$time, survival_test$cens),
  label = "Cox PH"
)

rsf_explainer <- explain(
  rsf_model,
  data = survival_test[, survival_predictors],
  y = Surv(survival_test$time, survival_test$cens),
  label = "Random Survival Forest"
)

# 5. Model Evaluation Metrics
cox_performance <- model_performance(cox_explainer)
rsf_performance <- model_performance(rsf_explainer)

print(cox_performance)
print(rsf_performance)

extract_scalar_metrics <- function(perf, model_name) {
  data.frame(
    Model = model_name,
    C_index = perf$result$`C-index`,
    Integrated_CD_AUC = perf$result$`Integrated C/D AUC`,
    Integrated_Brier_Score = perf$result$`Integrated Brier score`
  )
}

performance_table <- rbind(
  extract_scalar_metrics(cox_performance, "Cox PH"),
  extract_scalar_metrics(rsf_performance, "Random Survival Forest")
)

print(performance_table)

write.csv(
  performance_table,
  "tables/C02_model_performance.csv",
  row.names = FALSE
)

saveRDS(cox_model, "results/C02_cox_model.rds")
saveRDS(rsf_model, "results/C02_rsf_model.rds")
saveRDS(cox_explainer, "results/C02_cox_explainer.rds")
saveRDS(rsf_explainer, "results/C02_rsf_explainer.rds")