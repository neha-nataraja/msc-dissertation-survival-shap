# Load Packages

library(tidyverse)
library(caret)
library(randomForest)
library(kernelshap)
library(ggplot2)

# Load Dataset

data("mtcars")

# Data Preparation

set.seed(123)

train_index <- createDataPartition(
  mtcars$mpg,
  p = 0.8,
  list = FALSE
)

train_data <- mtcars[train_index, ]
test_data <- mtcars[-train_index, ]

# Fit Random Forest Model

rf_model <- randomForest(
  mpg ~ .,
  data = train_data,
  ntree = 500,
  importance = TRUE
)

# Select Observation

new_observation <- test_data[1, ]

# Calculate SHAP Values
predictors <- setdiff(names(train_data), "mpg")

shap_values <- kernelshap(
  object = rf_model,
  X = new_observation[, predictors],
  bg_X = train_data[, predictors]
)

# Extract SHAP Values

shap_results <- data.frame(
  Feature = colnames(shap_values$S),
  SHAP = as.numeric(shap_values$S[1, ])
)

print(shap_results)

# Sort SHAP Values

shap_results <- shap_results %>%
  arrange(desc(abs(SHAP)))

print(shap_results)

# Local SHAP Plot

shap_plot <- ggplot(
  shap_results,
  aes(
    x = reorder(Feature, abs(SHAP)),
    y = SHAP,
    fill = SHAP > 0
  )
) +
  geom_col() +
  scale_fill_manual(
    values = c(
      "TRUE" = "steelblue",
      "FALSE" = "tomato"
    )
  ) +
  coord_flip() +
  labs(
    title = "Local SHAP Explanation",
    x = "Feature",
    y = "SHAP Value"
  ) +
  theme_minimal() +
  theme(
    legend.position = "none"
  )

ggsave(
  filename = "figures/A03_local_shap.png",
  plot = shap_plot,
  width = 7,
  height = 5,
  dpi = 300
)

write.csv(
  shap_results,
  "tables/A03_local_shap.csv",
  row.names = FALSE
)
