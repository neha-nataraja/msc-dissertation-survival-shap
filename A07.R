library(tidyverse)
library(caret)
library(randomForest)
library(kernelshap)

# ------------------------------------------------------------
# Load Data
# ------------------------------------------------------------

data(mtcars)

set.seed(123)

train_index <- createDataPartition(
  mtcars$mpg,
  p = 0.8,
  list = FALSE
)

train_data <- mtcars[train_index, ]
test_data  <- mtcars[-train_index, ]

# ------------------------------------------------------------
# Create Overspecified Dataset
# ------------------------------------------------------------

set.seed(42)

train_over <- train_data
test_over  <- test_data

for(i in 1:5){
  
  train_over[[paste0("noise_", i)]] <- rnorm(nrow(train_over))
  
  test_over[[paste0("noise_", i)]]  <- rnorm(nrow(test_over))
  
}

predictors_base <- setdiff(names(train_data), "mpg")
predictors_over <- setdiff(names(train_over), "mpg")

# ------------------------------------------------------------
# Base Random Forest Stability
# ------------------------------------------------------------

base_results <- data.frame()

for(seed in 1:100){
  
  set.seed(seed)
  
  rf <- randomForest(
    mpg ~ .,
    data = train_data,
    ntree = 500
  )
  
  shap <- kernelshap(
    object = rf,
    X = test_data[, predictors_base],
    bg_X = train_data[, predictors_base]
  )
  
  base_results <- bind_rows(
    base_results,
    tibble(
      Model = "Base",
      Run = seed,
      Feature = colnames(shap$S),
      Mean_Abs_SHAP = colMeans(abs(shap$S))
    )
  )
  
}

# ------------------------------------------------------------
# Overspecified Random Forest Stability
# ------------------------------------------------------------

over_results <- data.frame()

for(seed in 1:100){
  
  set.seed(seed)
  
  rf <- randomForest(
    mpg ~ .,
    data = train_over,
    ntree = 500
  )
  
  shap <- kernelshap(
    object = rf,
    X = test_over[, predictors_over],
    bg_X = train_over[, predictors_over]
  )
  
  over_results <- bind_rows(
    over_results,
    tibble(
      Model = "Overspecified",
      Run = seed,
      Feature = colnames(shap$S),
      Mean_Abs_SHAP = colMeans(abs(shap$S))
    )
  )
  
}

# ------------------------------------------------------------
# Combine Results
# ------------------------------------------------------------

all_results <- bind_rows(
  base_results,
  over_results
)

# ------------------------------------------------------------
# Summary Table
# ------------------------------------------------------------

stability_summary <- all_results %>%
  group_by(Model, Feature) %>%
  summarise(
    Mean_Absolute_SHAP = mean(Mean_Abs_SHAP),
    SD_SHAP = sd(Mean_Abs_SHAP),
    .groups = "drop"
  ) %>%
  arrange(
    Model,
    desc(Mean_Absolute_SHAP)
  )

write.csv(
  stability_summary,
  "tables/A07_overspecified_stability.csv",
  row.names = FALSE
)

# ------------------------------------------------------------
# Stability Plot
# ------------------------------------------------------------

sd_summary <- all_results %>%
  group_by(Model, Feature) %>%
  summarise(
    SD_SHAP = sd(Mean_Abs_SHAP),
    .groups = "drop"
  )

p <- ggplot(
  sd_summary,
  aes(
    x = reorder(Feature, SD_SHAP),
    y = SD_SHAP,
    fill = Model
  )
) +
  geom_col(position = "dodge") +
  coord_flip() +
  scale_fill_manual(
    values = c(
      "Base" = "steelblue",
      "Overspecified" = "tomato"
    )
  ) +
  labs(
    title = "Comparison of SHAP Stability Between Base and Overspecified Random Forest Models",
    x = "Feature",
    y = "Standard Deviation of Mean Absolute SHAP Values"
  ) +
  theme_minimal()

ggsave(
  "figures/A07_overspecified_stability.png",
  plot = p,
  width = 9,
  height = 6,
  dpi = 300
)

# ------------------------------------------------------------
# Display Results
# ------------------------------------------------------------

print(stability_summary, n = Inf)

cat("\nA07 completed successfully.\n")