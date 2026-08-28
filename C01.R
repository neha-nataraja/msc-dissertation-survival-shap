library(tidyverse)
library(caret)
library(survival)
library(TH.data)

dir.create("tables", showWarnings = FALSE)
dir.create("figures", showWarnings = FALSE)
dir.create("results", showWarnings = FALSE)

data("GBSG2", package = "TH.data")

cat("GBSG2 loaded:", nrow(GBSG2), "rows,", ncol(GBSG2), "columns\n")
cat("Events:", sum(GBSG2$cens == 1), " Censored:", sum(GBSG2$cens == 0), "\n")

set.seed(123)

train_index <- createDataPartition(
  GBSG2$cens,
  p = 0.8,
  list = FALSE
)

survival_train <- GBSG2[train_index, ]
survival_test  <- GBSG2[-train_index, ]

cat("Training observations:", nrow(survival_train), "\n")
cat("Testing observations:", nrow(survival_test), "\n")
cat("Training events:", sum(survival_train$cens == 1), "\n")
cat("Testing events:", sum(survival_test$cens == 1), "\n")

survival_predictors <- c(
  "horTh", "age", "menostat", "tsize",
  "tgrade", "pnodes", "progrec", "estrec"
)

survival_formula <- as.formula(
  paste("Surv(time, cens) ~", paste(survival_predictors, collapse = " + "))
)

saveRDS(survival_train, "results/survival_train.rds")
saveRDS(survival_test, "results/survival_test.rds")
saveRDS(survival_predictors, "results/survival_predictors.rds")
saveRDS(survival_formula, "results/survival_formula.rds")