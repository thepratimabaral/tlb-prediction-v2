# TLB Risk Prediction using Weather Data (Nepal)

Turcicum Leaf Blight (TLB) is a fungal disease of maize, caused by *Exserohilum turcicum*. In this
project I tried to predict TLB risk for 62 districts of Nepal using daily weather data from 1981 to
2019. The idea is to know one day before if the weather will be good for the disease to spread, so
farmers can spray or take care in time.

This is the second version of my project. The first one is in
[tlb-diseases-prediction](https://github.com/thepratimabaral/tlb-diseases-prediction).

## About the label

I don't have real field data of TLB cases, so I made the label from weather. I used the rule from my
MSc thesis, based on Sharma (2023) and Radhajeyalakshmi et al. (2023). A day is counted as TLB
favourable if:

- humidity (RH) is more than 80%, and
- average temperature is between 18 and 25 °C

So the model is predicting favourable weather for the disease, not the actual disease.

About 9% of the days are favourable, so the data is imbalanced. Almost all of them are from June to
October.

**Correction:** an earlier version of this repo used by mistake the weather rule for potato late
blight (RH > 80% and minimum temperature < 10 °C), which is a different disease. That made the cold
mountain districts look like the highest risk. I changed it to the TLB rule above and ran everything
again. The results below are from the corrected version.

## What I changed from the first version

In my first notebook all the models got around 0.99 accuracy and F1. Later I found the reason: the
label was made from the weather of the same day, and that same weather was also given as features. So
the model was basically just learning my rule back. In this version I fixed that and some other things:

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
| XGBoost (tuned) | 0.869 | 0.878 | 0.874 | 0.950 | 0.993 |
| CatBoost | 0.872 | 0.870 | 0.871 | 0.947 | 0.993 |
| Random Forest | 0.869 | 0.872 | 0.871 | 0.947 | 0.992 |
| XGBoost | 0.846 | 0.893 | 0.869 | 0.947 | 0.992 |
| Decision Tree | 0.824 | 0.894 | 0.858 | 0.928 | 0.989 |
| Logistic Regression | 0.819 | 0.887 | 0.852 | 0.900 | 0.987 |
| Baseline (today = tomorrow) | 0.863 | 0.863 | 0.863 | 0.761 | 0.922 |

Tuned XGBoost gave the best result. But the baseline is very strong, because weather doesn't change
much from one day to the next. So the models are only a little better in F1 (0.874 vs 0.863). PR-AUC
is much better (0.76 vs 0.95), which means the predicted probabilities are more useful for ranking
risk than a simple yes/no.

The highest risk districts were in the mid-hills, like Sankhuwasabha, Dhankuta, Syangja, Okhaldhunga
and the Kathmandu valley, mostly from July to September. The high mountain districts have almost no
risk because they are too cold, and many Terai districts are often too hot (above 25 °C) in monsoon.

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

- the label is only based on a weather rule, it should be checked with real TLB records
- the rule doesn't know where maize is actually grown, so I should add maize area and elevation data
- some districts share the same latitude and longitude in the dataset (for example Kathmandu,
  Lalitpur, Dhading and Nuwakot), so they get exactly the same weather and the same risk
- I only tried 1 day ahead, would like to try 3 or 7 days ahead later
