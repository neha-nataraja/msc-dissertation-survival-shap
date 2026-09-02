# On the Stability of SHAP-Based Explanations for Black-Box Models

**Applications to motor insurance claim severity and survival analysis**

MSc dissertation, Data Science & Analytics, University of Leeds (2026)
Author: Neha Belagarahalli Nataraja · Supervisor: George Aivaliotis, School of Mathematics

## What this is

SHAP (SHapley Additive exPlanations) is close to the default choice for explaining
black-box model predictions, and is increasingly expected by regulators and governance
committees in general insurance. But SHAP's mathematical guarantees don't, on their own,
guarantee the explanation is *stable* — that refitting the same model with a different
random seed, growing more trees, or including a few uninformative predictors wouldn't
change which features SHAP says matter most.

This project stress-tests SHAP stability across three settings of increasing realism:

1. **Proof-of-concept** — a random forest predicting fuel economy (`mtcars`)
2. **Applied regression** — a two-part model of Australian motor insurance claim
   severity (`dataCar`, ~68k policies)
3. **Survival analysis** — breast-cancer recurrence-free survival (`GBSG2`), using
   the time-dependent **SurvSHAP(t)** extension to explain a Cox proportional hazards
   model and a random survival forest

In each setting, SHAP importance rankings are tested against three sources of
instability: **random seed**, **ensemble size** (number of trees), and
**overspecification** (adding predictors with no genuine signal).

## Key findings

- **Seed-to-seed variation is mild.** Kendall's coefficient of concordance was above
  0.98 in both applied settings (W = 0.989 insurance, W = 0.992 survival) — near-perfect
  rank agreement across refits.
- **Ensemble-size instability falls off predictably.** Standard deviation of SHAP
  importance dropped roughly fivefold from 50 to 1,000 trees, consistently across all
  three settings despite very different sample sizes and outcome types.
- **Overspecification is the real problem.** Adding just five random noise columns let
  a spurious predictor outrank a genuine one — a noise variable ranked *2nd* overall in
  the insurance model (ahead of vehicle value), and in the survival model, three of five
  injected noise columns outranked hormonal therapy status, a real clinical predictor.
  Weaker genuine predictors lost the most importance; the strongest was barely affected.
- **SHAP and GLM coefficients can disagree sharply** on which predictor matters most —
  a reminder that "feature importance" isn't a single, method-independent quantity.
- **SurvSHAP(t) reveals sign changes over time** that a static SHAP value can't:
  progesterone and estrogen receptor level start as risk-*increasing* early in
  follow-up and become risk-*decreasing* later, with only moderate rank agreement
  between early and late follow-up (Spearman ρ = 0.738).

**Practical takeaway:** SHAP is a genuinely useful diagnostic, but shouldn't be read as
ground truth without first checking model specification and ensemble size — directly
relevant for insurers and other regulated users relying on SHAP to justify model-driven
decisions.

## Tech stack

R 4.5 · `kernelshap` (tabular KernelSHAP) · `survex` (SurvSHAP(t)) · `randomForest` /
`ranger` · `survival` · `tidyverse` · `caret` · reproducible via `renv`

## Repository structure

Each script writes tables to `tables/`, figures to `figures/`, and fitted
model/explainer objects to `results/`, so later scripts reload intermediate objects
rather than refitting from scratch.

| Setting | Scripts | Covers |
|---|---|---|
| A — proof-of-concept (`mtcars`) | `A01`–`A07` | Model fit, local/global SHAP, seed stability, tree-count stability, overspecification |
| B — insurance claim severity (`dataCar`) | `B00`–`B06`, `Glm vs shap comparison.R` | Data prep, EDA, two-part GLM vs random forest, SHAP + efficiency check, seed/tree-count/overspecification stability, GLM-vs-SHAP ranking comparison |
| C — survival analysis (`GBSG2`) | `C01`–`C06` | Cox PH vs random survival forest, local/global SurvSHAP(t), seed/tree-count/overspecification stability |

Full dissertation write-up (methodology, statistical detail, limitations, and future
work) available on request.

## Note on AI use

As declared in the dissertation itself: Claude was used to assist with R code and
debugging, structuring chapters, drafting prose, and formatting the LaTeX source.
All analysis, interpretation, and conclusions are the author's own.
