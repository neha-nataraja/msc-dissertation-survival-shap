# ============================================================
# GLM Coefficients vs SHAP Ranking Comparison (Insurance Severity)
# ============================================================

library(tidyverse)

severity_glm <- readRDS("results/B02_severity_glm.rds")
global_shap <- read.csv("tables/B03_global_shap.csv")

glm_summary <- summary(severity_glm)$coefficients

glm_df <- data.frame(
  Feature = rownames(glm_summary),
  Coefficient = glm_summary[, "Estimate"],
  P_Value = glm_summary[, "Pr(>|t|)"]
) %>%
  filter(Feature != "(Intercept)")

print(glm_df)

glm_df$Base_Feature <- gsub("[A-Z0-9]+$", "", glm_df$Feature)

glm_feature_summary <- glm_df %>%
  group_by(Base_Feature) %>%
  summarise(Max_Abs_Coefficient = max(abs(Coefficient)), .groups = "drop") %>%
  arrange(desc(Max_Abs_Coefficient))

print(glm_feature_summary)

comparison <- global_shap %>%
  rename(Base_Feature = Feature) %>%
  left_join(glm_feature_summary, by = "Base_Feature") %>%
  arrange(desc(Mean_Absolute_SHAP)) %>%
  mutate(
    SHAP_Rank = row_number(),
    GLM_Rank = rank(-Max_Abs_Coefficient, na.last = "keep")
  )

print(comparison)

write.csv(
  comparison,
  "tables/B03_glm_vs_shap_comparison.csv",
  row.names = FALSE
)

cat("Comparison completed. Check Base_Feature column for any factor-name\n")
cat("stripping errors before using this table.\n")