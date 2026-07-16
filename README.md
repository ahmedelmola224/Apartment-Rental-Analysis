# What Determines How Much You Pay for an Apartment in the United States?
 
**Interactive Dashboard:** [dashboard](https://muratbek-kebtarum.github.io/ASDA_2025_Group_5_Portfolio/)
 

 
## Overview
 
This project investigates what drives apartment rent prices in the United States, using a dataset of 10,000 rental listings scraped from US property websites (September–December 2019), covering 51 states and 1,572 cities. The analysis combines exploratory data analysis, statistical hypothesis testing, dimensionality reduction, and predictive modeling to identify the key factors behind rent variation — and to build a model that predicts rent from listing attributes.
 
## Dataset
 
| Item | Description |
|---|---|
| Name | `apartments_for_rent_classified_10K` |
| Time period | September–December 2019 (~3.3 months) |
| Rows | 10,000 (raw) |
| Columns | 22 (raw) |
| Format | `.csv` (semicolon-delimited, cp1252 encoding) |
 
Each listing includes rent price, physical attributes (square footage, bedrooms, bathrooms), geographic data (city, state, GPS coordinates), a free-text amenities description, pet policy, and listing metadata.
 
## Methodology
 
### 1. Data Cleaning
- Dropped near-constant columns carrying almost no information (`fee`, `has_photo`, `is_studio`, `source`, `currency`, `category`) — each with a single value dominating over 90% of entries.
- Removed rows with missing or invalid values (nulls and zero-bathroom data-entry errors).
- Standardized mixed price periods (weekly/daily) to monthly rent.
- Filled missing `pets_allowed` values (42% null) with "Not specified" to retain rows.
- Applied the **IQR method** to remove price and square-footage outliers (615 price outliers, bounds $0–$2,814; 511 square-feet outliers, bounds 12–1,688 sq ft).
### 2. Exploratory Data Analysis
- Characterized the "typical" apartment: ~$1,200/month median rent, ~850 sq ft, 1-bed/1-bath dominant.
- Analyzed price vs. square footage (positive but noisy relationship) using scatterplots and hexbin density heatmaps.
- Investigated price-per-square-foot using LOWESS regression, revealing a "bulk value" effect — studios cost the most per square foot, 3-bedroom units offer the best value.
- Explored the "bedroom puzzle": studios cost *more* on average than 1-bedroom apartments, explained by their concentration in expensive urban cores (Manhattan, San Francisco, Boston).
- Tested statistical significance of rent differences across bedroom counts using **Welch's ANOVA** (robust to unequal variances, confirmed via Q-Q plots and boxplots) — F(5, 73.7) = 93.88, p < 0.001, η² = 0.051 — followed by a **Games-Howell post-hoc test**.
- Quantified geographic effects: state-level median rents (Massachusetts $2,000, California $1,994 vs. sub-$1,000 midwestern states) and city-level drivers (Boston $2,624, San Francisco $2,394).
- Checked temporal stability across the 4-month window (too short for seasonality conclusions).
### 3. Dimensionality Reduction
- Applied **Principal Component Analysis (PCA)** across 5 numeric features (square feet, bedrooms, bathrooms, latitude, longitude).
- PC1 (46.7% variance) captured "apartment size"; PC2 (22.5% variance) captured "geographic position" — confirming size and location as the two dominant latent drivers of rent.
### 4. Feature Engineering
- Extracted **15 binary amenity flags** via regex from the free-text `amenities` column (parking, pool, gym, hardwood floors, etc.).
- Created `pets_allowed_binary` and `is_luxury` (from listing title keywords).
- Built `state_avg_price` and `state_median_price` as target-encoded features (log-scale, trained on training data only to avoid leakage).
- Checked multicollinearity via correlation matrix and VIF before modeling.
### 5. Predictive Modeling
Four models were trained and compared on log-transformed price:
 
| Model | Test R² | RMSE ($) | MAE ($) | MAPE (%) | Features |
|---|---|---|---|---|---|
| **CatBoost** | **0.7225** | **261** | **185** | **15.3** | 22 |
| Random Forest (GridSearchCV) | 0.6928 | 274 | 193 | 16.2 | 22 |
| OLS Reduced (backward elimination) | 0.5243 | 341 | 252 | 21.0 | 10 |
| OLS Full | 0.5240 | 342 | 252 | 21.0 | 22 |
 
- **OLS (full & reduced):** Baseline linear regression; state average rent and square footage were the dominant standardized coefficients. Backward elimination reduced 22 features to 10 with no loss in test performance.
- **Random Forest:** Tuned over 24 hyperparameter combinations with 5-fold cross-validation (best: 300 trees, max depth 16). Captured non-linear interactions missed by OLS, with moderate overfitting (R² gap of 0.156).
- **CatBoost:** Best-performing model, using 2,000 iterations, learning rate 0.05, depth 7, and early stopping. Chosen for its native handling of categorical features, strong tabular-data performance, and ordered boosting, which reduces overfitting relative to traditional gradient boosting.
## Key Findings
 
1. **Location is the strongest predictor of rent** — state and city-level geography outweigh most physical attributes.
2. **Size matters, with diminishing returns** — larger apartments cost more in total but less per square foot, with 3-bedroom units offering the best value.
3. **Most amenities are marketing, not value** — of 15 amenity features, only elevators, parking, and hardwood floors showed consistent statistical significance once location and size were controlled for.
4. **Non-linear models substantially outperform linear ones** — CatBoost improved test R² from 0.524 (OLS) to 0.723, showing that rent pricing involves meaningful feature interactions.
5. **About 28% of rent variation remains unexplained**, likely driven by factors absent from the dataset: building age, interior quality, transit proximity, neighborhood reputation, and lease terms.
## Limitations
 
- **Short time window:** Data spans only ~3.3 months (Sep–Dec 2019), preventing analysis of seasonality, year-over-year trends, or external shocks (e.g., COVID-19).
- **Geographic sampling bias:** Listings are not a representative sample of the US rental market; some states/cities are over-represented.
- **Missing variables:** Building age, interior condition, transit access, safety, and lease terms are not captured.
- **Regex-based amenity parsing** may miss amenities due to misspellings or unusual phrasing.
- **Tree-model overfitting:** R² gaps of 0.13–0.16 suggest limited generalization to other time periods or regions.
- **Target-encoding noise:** States with few listings produce less reliable encoded values.
## Tools & Reproducibility
 
- **Stack:** Python 3.13, pandas, NumPy, Matplotlib, Seaborn, Plotly, SciPy, statsmodels, pingouin, scikit-learn, CatBoost
- **Environment:** Jupyter Notebook (VS Code)
- **Reproducibility:** All analysis code is in `apartments_for_rent.ipynb`; dataset loads from HuggingFace; random seed = 42 for all stochastic operations.
