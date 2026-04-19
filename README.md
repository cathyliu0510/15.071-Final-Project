# 15.071 Supply Chain Risk & Logistics — Project README

**15.071 Final Project** | Avery Reischer-Craft · Carla Choueifaty · Cathy Liu  
**Dataset:** Global Supply Chain Risk & Logistics 2024–2026 (5,000 shipments, 14 columns)

---

## What This Project Does

This project predicts two supply chain outcomes from international shipment data. The first target, `Disruption_Occurred`, is a binary variable indicating whether a shipment was delayed or cancelled, and is treated as a classification problem. The second target, `Lead_Time_Days`, is a continuous variable representing actual delivery time, and is treated as a regression problem. The models built are Logistic Regression and CART for classification, and Linear Regression and CART for regression.

---

## What the Notebook Does

The notebook `supply_chain_eda_preprocessing.ipynb` handles all exploratory analysis and data preparation. It should be run first, before any modeling. The notebook expects the raw CSV file `global_supply_chain_risk_2026.csv` to be placed in the same directory. Running all cells top to bottom produces two outputs: a folder called `plots/` containing 18 saved EDA figures, and a file called `preprocessed_data.pkl` containing all the data splits and transformers needed for modeling.

### Section 0 — Imports and Setup

Loads all required libraries and defines shared visual settings used across every plot. Also creates the `plots/` output directory.

### Section 1 — Load Data

Reads the CSV and performs a basic audit. The dataset has 5,000 rows and 14 columns with no missing values. There are 5 numeric features (`Distance_km`, `Weight_MT`, `Fuel_Price_Index`, `Geopolitical_Risk_Score`, `Carrier_Reliability_Score`), 5 categorical features (`Origin_Port`, `Destination_Port`, `Transport_Mode`, `Product_Category`, `Weather_Condition`), two non-informative ID columns (`Shipment_ID`, `Date`), and the two target variables.

### Section 2 — Exploratory Data Analysis

The EDA produces 18 plots, all saved to the `plots/` folder. T

### Section 3 — Preprocessing

The preprocessing pipeline prepares two distinct versions of the feature matrix for modeling, because different model types have different requirements.

First, `Shipment_ID` and `Date` are dropped since they carry no predictive signal. The two targets are then separated from the features.

**Encoding.** For Logistic Regression and Linear Regression, the categorical features are one-hot encoded using `pd.get_dummies` with `drop_first=True`, which removes one dummy per category to avoid multicollinearity. This produces a 31-column feature matrix. For CART, the categorical features are instead label-encoded into integer codes, since decision trees can split on integers natively and do not require dummy variables. This produces a 10-column feature matrix.

**Train/test split.** The data is split 80% train and 20% test. For the classification task, the split is stratified on `Disruption_Occurred` to ensure both sets have the same class balance (61.3% disruption in both). The same random seed (42) is used throughout for reproducibility.

**Scaling.** A `StandardScaler` is applied to the 5 numeric columns for the Logistic Regression and Linear Regression versions of the data; these models are sensitive to feature magnitude. Critically, the scaler is fitted on the training set only and then applied to the test set, which prevents data leakage. CART inputs are left unscaled since trees are scale-invariant.

Everything is then saved into `preprocessed_data.pkl`.

---

## How to Use the Preprocessed Data

Load the pickle file at the top of any modeling notebook:

```python
import pickle
with open('preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)
```

### For Logistic Regression

Use `data['X_cls_train_scaled']` and `data['X_cls_test_scaled']` as features, with `data['y_cls_train']` and `data['y_cls_test']` as the target. These are the scaled, one-hot encoded matrices with 31 features. Feature names are available in `data['ohe_feature_names']` and are useful for interpreting model coefficients.

### For Linear Regression

Use `data['X_reg_train_scaled']` and `data['X_reg_test_scaled']` as features, with `data['y_reg_train']` and `data['y_reg_test']` as the target. These are also scaled one-hot encoded matrices with 31 features. Consider Ridge or Lasso variants for regularization, tuning the penalty strength via cross-validation on the training set.

### For CART — Classification

Use `data['X_cls_train_le']` and `data['X_cls_test_le']` as features, with `data['y_cls_train']` and `data['y_cls_test']` as the target. These are the label-encoded, unscaled matrices with 10 features. Tune `max_depth` and `min_samples_leaf` using cross-validation on the training set to control tree complexity and prevent overfitting.

### For CART — Regression

Use `data['X_reg_train_le']` and `data['X_reg_test_le']` as features, with `data['y_reg_train']` and `data['y_reg_test']` as the target. These are the same label-encoded, unscaled matrices with 10 features. Apply the same cross-validation tuning strategy as the classification tree.

### Recovering Category Labels

The label encoders for each categorical column are stored in `data['le_dict']`. These are useful when interpreting CART splits — for example, to find out which integer code maps to which port or transport mode. The reference categories dropped during one-hot encoding are Antwerp (ports), Air (transport mode), Automotive (product category), and Clear (weather condition).
