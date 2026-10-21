# TLB Risk Prediction using Weather Data (Nepal)

In this project I tried to predict late blight (TLB) risk in potato/tomato for 62 districts of Nepal
using daily weather data from 1981 to 2019. The idea is to know one day before if the weather will
be good for the disease to spread, so farmers can spray or take care in time.

This is the second version of my project. The first one is in
[tlb-diseases-prediction](https://github.com/thepratimabaral/tlb-diseases-prediction).

## About the label

I don't have real field data of blight cases, so I made the label from weather. A day is counted as
TLB favourable if humidity (RH) is more than 80% and minimum temperature is less than 10 °C.
So the model is predicting favourable weather for the disease, not the actual disease.

Only around 3% of the days are favourable, so the data is very imbalanced.

## What I changed from the first version

In my first notebook all the models got around 0.99 accuracy and F1. Later I found the reason: the
label was made from RH and min temperature of the same day, and RH was also given as a feature. So the
model was basically just learning my rule back. In this version I fixed that and some other things:

- the model now predicts the next day's condition using today's and past few days' weather
- train/test split is done by year (train 1981-2011, validation 2012-2015, test 2016-2019),
  not random, because nearby days are almost the same and random split gives too good results
- I removed SMOTE and the scaling on the whole dataset, now I use class weights and the scaler is
  only fitted on training data
- added a simple baseline "tomorrow will be same as today" to compare with
- added lag and rolling features (last 3 and 7 days), SHAP plot and district risk map
- cleaned the notebook (removed repeated cells and Kaggle paths)

## Results

Test set is 2016-2019.

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| XGBoost (tuned) | 0.703 | 0.791 | 0.744 | 0.841 | 0.988 |
| CatBoost | 0.674 | 0.812 | 0.737 | 0.837 | 0.988 |
| Random Forest | 0.690 | 0.789 | 0.736 | 0.836 | 0.987 |
| XGBoost | 0.733 | 0.752 | 0.742 | 0.835 | 0.987 |
| Decision Tree | 0.712 | 0.755 | 0.733 | 0.783 | 0.967 |
| Logistic Regression | 0.695 | 0.729 | 0.711 | 0.740 | 0.980 |
| Baseline (today = tomorrow) | 0.717 | 0.717 | 0.717 | 0.528 | 0.851 |

Tuned XGBoost gave the best result. The baseline is actually quite strong because weather doesn't
change much from one day to the next, so the models are only a little better in F1. But PR-AUC is
much better (0.53 vs 0.84), which means the predicted probabilities are more useful for ranking risk.

The highest risk districts were Manang, Mustang, Rasuwa, Dolpa and Mugu, mostly in monsoon months.
These are the high mountain districts where it stays cold and humid.

## Files

- `tlb_prediction.ipynb` - main notebook (already run so you can see the outputs)
- `outputs/` - plots and result tables
- `data/` - put `dailyweather.csv` here (not uploaded because it is more than 100 MB)
- `requirements.txt`

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook tlb_prediction.ipynb
```

It takes around 4-5 minutes to run everything. On Mac, XGBoost may need `brew install libomp`.

## Limitations

- the label is only based on weather rule, it should be checked with real disease records
- no elevation or crop area data used
- I only tried 1 day ahead, would like to try 3 or 7 days ahead later
