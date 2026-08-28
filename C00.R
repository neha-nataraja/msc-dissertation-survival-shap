spike_start <- Sys.time()

required_packages <- c("survival", "TH.data", "survex")

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat("Installing", pkg, "...\n")
    install.packages(pkg)
  }
}

library(survival)
library(TH.data)
library(survex)

data("GBSG2", package = "TH.data")

cat("GBSG2 loaded:", nrow(GBSG2), "rows,", ncol(GBSG2), "columns\n")
str(GBSG2)

cat("\nFitting Cox model...\n")

cox_start <- Sys.time()

cox_model <- coxph(
  Surv(time, cens) ~ horTh + age + menostat + tsize + tgrade +
    pnodes + progrec + estrec,
  data = GBSG2,
  x = TRUE  # needed for survex
)

cox_time <- Sys.time() - cox_start
cat("Cox model fitted in:", round(as.numeric(cox_time), 2), "sec\n")
print(summary(cox_model))

cat("\nBuilding survex explainer...\n")

explainer_start <- Sys.time()

cox_explainer <- explain(
  cox_model,
  data = GBSG2[, c("horTh", "age", "menostat", "tsize", "tgrade",
                   "pnodes", "progrec", "estrec")],
  y = Surv(GBSG2$time, GBSG2$cens)
)

explainer_time <- Sys.time() - explainer_start
cat("Explainer built in:", round(as.numeric(explainer_time), 2), "sec\n")

cat("\nComputing SurvSHAP(t) for one observation...\n")

shap_start <- Sys.time()

new_obs <- GBSG2[1, c("horTh", "age", "menostat", "tsize", "tgrade",
                      "pnodes", "progrec", "estrec")]

survshap_result <- predict_parts(
  cox_explainer,
  new_observation = new_obs,
  type = "survshap"
)

shap_time <- Sys.time() - shap_start
cat("SurvSHAP(t) computed in:", round(as.numeric(shap_time), 2), "sec\n")

print(survshap_result)

plot(survshap_result)

total_time <- Sys.time() - spike_start
cat("TOTAL SPIKE TIME:", round(as.numeric(total_time, units = "mins"), 2), "minutes\n")
