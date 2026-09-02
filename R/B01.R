# ============================================================
# B01: Insurance Exploratory Data Analysis
# ============================================================

library(tidyverse)
library(corrplot)

# ------------------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------------------

insurance_data <- readRDS(
  "results/insurance_data.rds"
)

severity_data <- readRDS(
  "results/severity_data.rds"
)

# ------------------------------------------------------------
# 2. Summary Statistics
# ------------------------------------------------------------

summary(insurance_data)

# ------------------------------------------------------------
# 3. Missing Values
# ------------------------------------------------------------

missing_values <- data.frame(
  Variable = names(insurance_data),
  Missing = colSums(is.na(insurance_data))
)

write.csv(
  missing_values,
  "tables/B01_missing_values.csv",
  row.names = FALSE
)

# ------------------------------------------------------------
# 4. Claim Occurrence
# ------------------------------------------------------------

claim_counts <- insurance_data %>%
  count(clm) %>%
  mutate(
    Claim_Status = ifelse(
      clm == 1,
      "Claim",
      "No claim"
    ),
    Percentage = n / sum(n) * 100
  )

print(claim_counts)

write.csv(
  claim_counts,
  "tables/B01_claim_occurrence.csv",
  row.names = FALSE
)

# ------------------------------------------------------------
# 5. Distribution of All Claim Costs
# ------------------------------------------------------------

p1 <- ggplot(
  insurance_data,
  aes(x = claimcst0)
) +
  geom_histogram(
    bins = 40,
    fill = "steelblue",
    colour = "black"
  ) +
  labs(
    title = "Distribution of Insurance Claim Costs",
    x = "Claim Cost",
    y = "Frequency"
  ) +
  theme_minimal()

ggsave(
  "figures/B01_claim_cost_distribution.png",
  p1,
  width = 8,
  height = 5,
  dpi = 300
)

# ------------------------------------------------------------
# 6. Positive Claim Cost Distribution
# ------------------------------------------------------------

p2 <- ggplot(
  severity_data,
  aes(x = claimcst0)
) +
  geom_histogram(
    bins = 40,
    fill = "steelblue",
    colour = "black"
  ) +
  labs(
    title = "Distribution of Positive Insurance Claim Costs",
    x = "Claim Cost",
    y = "Frequency"
  ) +
  theme_minimal()

ggsave(
  "figures/B01_positive_claim_cost_distribution.png",
  p2,
  width = 8,
  height = 5,
  dpi = 300
)

# ------------------------------------------------------------
# 7. Log-Transformed Positive Claim Costs
# ------------------------------------------------------------

p3 <- ggplot(
  severity_data,
  aes(x = log(claimcst0))
) +
  geom_histogram(
    bins = 40,
    fill = "steelblue",
    colour = "black"
  ) +
  labs(
    title = "Distribution of Log-Transformed Positive Claim Costs",
    x = "Log Claim Cost",
    y = "Frequency"
  ) +
  theme_minimal()

ggsave(
  "figures/B01_log_claim_cost_distribution.png",
  p3,
  width = 8,
  height = 5,
  dpi = 300
)

# ------------------------------------------------------------
# 8. Vehicle Value vs Positive Claim Cost
# ------------------------------------------------------------

p4 <- ggplot(
  severity_data,
  aes(
    x = veh_value,
    y = claimcst0
  )
) +
  geom_point(alpha = 0.4) +
  labs(
    title = "Vehicle Value vs Positive Claim Cost",
    x = "Vehicle Value",
    y = "Claim Cost"
  ) +
  theme_minimal()

ggsave(
  "figures/B01_vehicle_value.png",
  p4,
  width = 8,
  height = 5,
  dpi = 300
)

# ------------------------------------------------------------
# 9. Vehicle Age
# ------------------------------------------------------------

p5 <- ggplot(
  severity_data,
  aes(
    x = factor(veh_age),
    y = claimcst0
  )
) +
  geom_boxplot() +
  labs(
    title = "Positive Claim Cost by Vehicle Age",
    x = "Vehicle Age",
    y = "Claim Cost"
  ) +
  theme_minimal()

ggsave(
  "figures/B01_vehicle_age.png",
  p5,
  width = 8,
  height = 5,
  dpi = 300
)

# ------------------------------------------------------------
# 10. Gender
# ------------------------------------------------------------

p6 <- ggplot(
  severity_data,
  aes(
    x = gender,
    y = claimcst0
  )
) +
  geom_boxplot() +
  labs(
    title = "Positive Claim Cost by Gender",
    x = "Gender",
    y = "Claim Cost"
  ) +
  theme_minimal()

ggsave(
  "figures/B01_gender.png",
  p6,
  width = 8,
  height = 5,
  dpi = 300
)

# ------------------------------------------------------------
# 11. Vehicle Body Type
# ------------------------------------------------------------

p7 <- ggplot(
  severity_data,
  aes(
    x = veh_body,
    y = claimcst0
  )
) +
  geom_boxplot() +
  theme_minimal() +
  theme(
    axis.text.x = element_text(
      angle = 45,
      hjust = 1
    )
  ) +
  labs(
    title = "Positive Claim Cost by Vehicle Body Type",
    x = "Vehicle Body",
    y = "Claim Cost"
  )

ggsave(
  "figures/B01_vehicle_body.png",
  p7,
  width = 10,
  height = 6,
  dpi = 300
)

# ------------------------------------------------------------
# 12. Correlation Matrix
# ------------------------------------------------------------

numeric_data <- severity_data %>%
  select(where(is.numeric)) %>%
  select(
    -claimcst0,
    -clm
  )

corr_matrix <- cor(
  numeric_data,
  use = "pairwise.complete.obs"
)

png(
  "figures/B01_correlation_matrix.png",
  width = 800,
  height = 800
)

corrplot(
  corr_matrix,
  method = "color",
  tl.cex = 0.8
)

dev.off()

cat("B01 completed successfully.\n")