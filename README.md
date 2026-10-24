# Calories Burned Prediction

Predicting calories burned during exercise from physiological and personal
features, comparing 8 regression models with cross-validation and tuning.

## Dataset
- 15,000 records: `exercise.csv` (Gender, Age, Height, Weight, Duration,
  Heart_Rate, Body_Temp) merged with `calories.csv` on `User_ID`.
- No missing values, no duplicates. Target: `Calories`.

## Pipeline
1. Merge and clean data, encode Gender, drop `User_ID`
2. EDA: distributions, correlation matrix, feature vs target plots
3. Feature engineering: BMI, Heart_Rate x Duration, Body_Temp x Duration
4. 80/20 train/test split (random_state=42)
5. Train and compare 8 models, 5-fold cross-validation
6. Hyperparameter tuning of the top models

## Results (test set)
| Model | R2 | MAE | RMSE |
|---|---|---|---|
| Polynomial (deg 3) + Ridge [tuned] | 1.0000 | 0.255 | 0.294 |
| SVR (RBF) [tuned] | 1.0000 | 0.261 | 0.308 |
| CatBoost [tuned] | 0.9998 | 0.524 | 0.785 |
| LightGBM | 0.9996 | 0.896 | 1.343 |
| XGBoost | 0.9996 | 0.957 | 1.343 |
| Polynomial (deg 2) + Ridge | 0.9987 | 1.717 | 2.329 |
| Random Forest | 0.9983 | 1.689 | 2.645 |
| Linear Regression | 0.9845 | 5.819 | 7.897 |

## Key findings
- Duration (0.96), Heart_Rate (0.90) and Body_Temp (0.82) correlate most with Calories.
- Relationships are non-linear, so the linear baseline is about 27x worse than Polynomial deg 3.
- Remaining error (~0.29 RMSE) matches the rounding of the integer target, so the best models are near the floor.
- Data looks synthetic (smooth formula); real sensor data would be noisier.

## Run
```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
Then run `notebooks/01_eda_and_cleaning.ipynb` and `notebooks/02_models_comparison.ipynb`.