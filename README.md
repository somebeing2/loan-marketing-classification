# Predictive Analytics for Marketing: Personal Loan Campaign Response

**CIA Project — Marketing Analytics**

## Business Problem
A bank's marketing team wants to run a targeted campaign to convert existing
liability (depositor) customers into personal loan customers. Blanket
campaigns are expensive and historically convert only a small fraction of
customers. This project builds a predictive model that identifies which
customers are most likely to accept a personal loan offer, so the marketing
team can target them specifically  improving campaign conversion rate and
reducing cost per acquisition.

## Dataset
- **File:** `Bank_Personal_Loan_Modelling.csv`
- **Size:** 5,000 customers, 14 attributes
- **Target variable:** `Personal Loan` (1 = accepted the loan offer in a
  past campaign, 0 = did not)
- **Key features:** Age, Experience, Income, Family size, CCAvg (avg. credit
  card spend), Education, Mortgage, Securities Account, CD Account, Online
  banking usage, Credit Card ownership

## Approach
1. **Data cleaning** — dropped non-predictive identifiers (`ID`, `ZIP Code`);
   corrected a small number of negative values in `Experience`.
2. **EDA** — class balance, correlation heatmap, income/education vs. loan
   acceptance.
3. **Train/test split** — 75/25, stratified on the target (class imbalance
   is ~90/10).
4. **Feature scaling** — `StandardScaler`, used for Logistic Regression and
   KNN (distance/gradient-based models); not needed for the tree models.
5. **Models trained** (each tuned via 5-fold `GridSearchCV`, optimizing
   F1-score):
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - K-Nearest Neighbors (KNN)
6. **Evaluation** — Accuracy, Precision, Recall, F1-Score, ROC-AUC,
   confusion matrices, ROC curves. F1/ROC-AUC are prioritized over raw
   accuracy because of the class imbalance.
7. **Leaderboard** — models ranked by F1-Score to identify the best
   performer.

## Results — Model Comparison Leaderboard

| Rank | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|---|
| 1 | Random Forest | 0.989 | 0.965 | 0.917 | 0.940 | 0.999 |
| 2 | Decision Tree | 0.986 | 0.955 | 0.892 | 0.922 | 0.997 |
| 3 | KNN | 0.960 | 0.949 | 0.617 | 0.748 | 0.923 |
| 4 | Logistic Regression | 0.951 | 0.817 | 0.633 | 0.714 | 0.963 |

**Best model: Random Forest.** Loan acceptance depends on non-linear
interactions between features (e.g. income × education × CD account
ownership), which tree-based models capture better than a linear model
(Logistic Regression) or a raw-distance model (KNN).

## Business Insight (Feature Importance)
Income, Education, Family size, and CD Account ownership are the strongest
predictors of loan acceptance  see `outputs/07_feature_importance.png`.
**Recommendation:** the marketing team should prioritize customers with
higher income, higher education levels, and existing CD accounts for the
next personal loan campaign.

## Repository Structure
```
├── Bank_Personal_Loan_Modelling.csv          # dataset
├── loan_marketing_classification_colab.ipynb # full source code (Colab notebook)
├── leaderboard.csv                           # model comparison results
├── outputs/                                  # EDA plots, confusion matrices, ROC curves
└── README.md
```

## How to Run
1. Open `loan_marketing_classification_colab.ipynb` in
   [Google Colab](https://colab.research.google.com).
2. Run all cells top to bottom (**Runtime → Run all**).
3. When prompted, upload `Bank_Personal_Loan_Modelling.csv`.
4. Outputs (plots + leaderboard) are generated automatically and can be
   downloaded via the last cell.

## Tools & Libraries
Python, pandas, numpy, scikit-learn, matplotlib, seaborn

## Author
Kevin — MBA Business Analytics, CHRIST (Deemed to be University), Bengaluru
