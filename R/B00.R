# ============================================================
# B00: Insurance Data Preparation (Two-Part Structure)
# Produces insurance_data.rds (full population, for the occurrence
# model) and severity_data.rds (claimants only, for the severity
# model) - both loaded directly by B02 onward.
#
# Uses dplyr:: namespacing throughout. If Hmisc has been loaded in
# this R session (e.g. from earlier survex/C-series work), it masks
# dplyr::select()/summarise() with incompatible versions - this
# caused a real error earlier in this project. Namespacing avoids
# it regardless of what else is attached. If starting fresh, it is
# still good practice to Session -> Restart R before running B00.
# ============================================================

library(tidyverse)
library(insuranceData)

dir.create("data", showWarnings = FALSE)
dir.create("results", showWarnings = FALSE)
dir.create("tables", showWarnings = FALSE)

# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------

data(dataCar)

cat("Original dataset dimensions:\n")
print(dim(dataCar))
cat("\nOriginal variables:\n")
print(names(dataCar))

# ------------------------------------------------------------
# 2. Inspect claim occurrence and claim cost
# ------------------------------------------------------------

claim_summary <- dataCar %>%
  dplyr::group_by(clm) %>%
  dplyr::summarise(
    n = dplyr::n(),
    zero_claim_cost = sum(claimcst0 == 0, na.rm = TRUE),
    positive_claim_cost = sum(claimcst0 > 0, na.rm = TRUE),
    mean_claim_cost = mean(claimcst0, na.rm = TRUE),
    median_claim_cost = median(claimcst0, na.rm = TRUE),
    .groups = "drop"
  )

print(claim_summary)

write.csv(
  claim_summary,
  "tables/B00_claim_summary.csv",
  row.names = FALSE
)

# ------------------------------------------------------------
# 3. Verify relationship between clm and claimcst0
# ------------------------------------------------------------

claim_zero_check <- table(
  dataCar$clm,
  dataCar$claimcst0 == 0,
  useNA = "ifany"
)

print(claim_zero_check)

write.csv(
  as.data.frame(claim_zero_check),
  "tables/B00_zero_claim_check.csv",
  row.names = FALSE
)

# ------------------------------------------------------------
# 4. Missing values
# ------------------------------------------------------------

missing_values <- data.frame(
  Variable = names(dataCar),
  Missing = colSums(is.na(dataCar))
)

print(missing_values)

write.csv(
  missing_values,
  "tables/B00_missing_values.csv",
  row.names = FALSE
)

# ------------------------------------------------------------
# 5. Build insurance_data: full population, for the occurrence
# model. numclaims and X_OBSTAT_ are dropped - numclaims because
# it represents claim-count information generated jointly with
# the outcome and would not be available when predicting future
# claim costs; X_OBSTAT_ is a constant administrative field.
# ------------------------------------------------------------

insurance_data <- dataCar %>%
  dplyr::select(-numclaims, -X_OBSTAT_)

cat("\ninsurance_data dimensions (full population):\n")
print(dim(insurance_data))
cat("\ninsurance_data variables:\n")
print(names(insurance_data))

# ------------------------------------------------------------
# 6. Build severity_data: claimants only, for the severity model.
# Same columns as insurance_data - the occurrence indicator clm
# is retained here (it will read as constant within this subset,
# and is explicitly excluded again at the predictor-definition
# step in B02/B03, not here).
# ------------------------------------------------------------

severity_data <- insurance_data %>%
  dplyr::filter(clm == 1, claimcst0 > 0)

cat("\nseverity_data dimensions (claimants only):\n")
print(dim(severity_data))

cat("\nClaim cost summary (severity_data):\n")
print(summary(severity_data$claimcst0))

# ------------------------------------------------------------
# 7. Save both datasets
# ------------------------------------------------------------

saveRDS(insurance_data, "results/insurance_data.rds")
saveRDS(severity_data,  "results/severity_data.rds")

write.csv(insurance_data, "data/insurance_data.csv", row.names = FALSE)
write.csv(severity_data,  "data/severity_data.csv",  row.names = FALSE)

cat("\nB00 completed successfully.\n")
cat("Produced: results/insurance_data.rds, results/severity_data.rds\n")