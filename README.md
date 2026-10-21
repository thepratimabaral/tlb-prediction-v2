# TLB risk prediction from daily weather (Nepal)

Predicting next-day **late blight (TLB) favourable conditions** for 62 districts of Nepal using
daily weather data from 1981 to 2019.

This is the cleaned version of my first notebook (`tlb-diseases-prediction`).

## What changed from the first version

- **Target is the next day.** Before, the label was made from same-day `RH_2m` and `MinTemp_2m` and
  `RH_2m` was also a feature, so the models just learned the rule back (~0.99 on everything).
  Now the model predicts tomorrow's condition from today's and past weather.
- **Split by year** (train 1981-2011, validation 2012-2015, test 2016-2019) instead of a random
  split, so no future days end up in training.
- **No scaling / SMOTE on the full data.** Scaling is inside the pipeline, and class imbalance
  is handled with class weights.
- **Persistence baseline** ("tomorrow = today") added, so the ML results can be compared fairly.
- Threshold picked on the validation set, metrics include PR-AUC (better for 3% positives).
- Removed duplicate cells, Kaggle paths and cells that depended on variables from other runs.
- Added SHAP and a district risk map that only uses the lat/lon already in the data (no geojson needed).

## Label

A day is TLB-favourable when **RH > 80 %** and **minimum temperature < 10 °C**.
This is a weather-based rule, not field records of the disease.

## Files

```
tlb_prediction.ipynb   main notebook (already run, outputs saved)
data/dailyweather.csv  daily weather, 62 districts (not on github, too big)
outputs/               figures, result tables and the saved model
requirements.txt
```

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook tlb_prediction.ipynb
```

Put `dailyweather.csv` inside the `data/` folder first. Full run takes about 4-5 min on a laptop.

On Mac, XGBoost needs the OpenMP library (`brew install libomp`).

## Results (test set 2016-2019)

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| XGBoost (tuned) | 0.703 | 0.791 | 0.744 | 0.841 | 0.988 |
| CatBoost | 0.674 | 0.812 | 0.737 | 0.837 | 0.988 |
| Random Forest | 0.690 | 0.789 | 0.736 | 0.836 | 0.987 |
| XGBoost | 0.733 | 0.752 | 0.742 | 0.835 | 0.987 |
| Decision Tree | 0.712 | 0.755 | 0.733 | 0.783 | 0.967 |
| Logistic Regression | 0.695 | 0.729 | 0.711 | 0.740 | 0.980 |
| Persistence (today = tomorrow) | 0.717 | 0.717 | 0.717 | 0.528 | 0.851 |

Best model: **tuned XGBoost**. Top risk districts: Manang, Mustang, Rasuwa, Dolpa, Mugu (mostly in the monsoon).
