#  California House Price Prediction

A machine learning project that predicts California median house values using the classic California Housing dataset. The project covers the full ML workflow — from exploratory data analysis to a production-ready training and inference pipeline.

---

##  Overview

Given block-level census data (location, income, housing age, etc.), this model predicts the **median house value** for a California district. The project is split into two parts:

- **`Analyzing_Data.ipynb`** — EDA, feature exploration, correlation analysis, and preprocessing experiments.
- **`main.py`** — Production pipeline: trains the model on first run, runs inference on subsequent runs, and saves all outputs to CSV.

---

##  Project Structure

```
├── housing.csv               # Raw dataset (California Housing)
├── main.py                   # Train / Inference script
├── Analyzing_Data.ipynb      # EDA and preprocessing notebook
├── model.pkl                 # Saved trained model (generated on first run)
├── pipeline.pkl              # Saved preprocessing pipeline (generated on first run)
├── input.csv                 # Test set features (generated on first run)
├── True_labels.csv           # Test set ground truth labels (generated on first run)
└── output.csv                # Predictions on test set (generated on inference run)
```

---

## Dataset

**California Housing Dataset** — derived from the 1990 U.S. Census.

| Feature | Description |
|---|---|
| `longitude` | Geographic coordinate |
| `latitude` | Geographic coordinate |
| `housing_median_age` | Median age of houses in block |
| `total_rooms` | Total rooms in block |
| `total_bedrooms` | Total bedrooms in block (has missing values) |
| `population` | Block population |
| `households` | Number of households in block |
| `median_income` | Median income (scaled, in tens of thousands) |
| `ocean_proximity` | Categorical — distance from ocean |
| `median_house_value` | **Target variable** |

- 20,640 total samples
- 1 categorical feature (`ocean_proximity`): `<1H OCEAN`, `INLAND`, `NEAR OCEAN`, `NEAR BAY`, `ISLAND`
- `total_bedrooms` has 207 missing values → handled via median imputation

---

##  ML Pipeline

### Train/Test Split
- **Stratified Shuffle Split** (80/20) based on `median_income` buckets — ensures income distribution is preserved in both sets.

### Preprocessing (`ColumnTransformer`)

**Numerical features:**
1. `SimpleImputer` (median strategy) — fills missing `total_bedrooms`
2. `StandardScaler` — standardizes all numerical columns

**Categorical features:**
1. `OneHotEncoder` (with `handle_unknown='ignore'`) — encodes `ocean_proximity`

### Model
- **Random Forest Regressor** (`sklearn`, `random_state=42`)
- No hyperparameter tuning applied (plug-and-play location in code for `GridSearchCV` / `RandomizedSearchCV`)

### Evaluation
- **MAPE (Mean Absolute Percentage Error)** — reported on the held-out test set

---

##  How to Run

### 1. Install dependencies

```bash
pip install pandas numpy scikit-learn joblib
```

### 2. First run — Train the model

Make sure `housing.csv` is in the same directory, then:

```bash
python main.py
```

This will:
- Train the Random Forest model
- Save `model.pkl` and `pipeline.pkl`
- Save `input.csv` and `True_labels.csv` (test set)
- Print: `Model is trained. Congrats!!!`

### 3. Subsequent runs — Inference

Run the same command again:

```bash
python main.py
```

This will:
- Load the saved model and pipeline
- Run predictions on `input.csv`
- Save results to `output.csv`
- Print MAPE on the test set

---

##  Results

| Metric | Value |
|---|---|
| MAPE | ~17–19% (typical for this dataset with default RF) |

> Results may vary slightly depending on the scikit-learn version.

---

##  EDA Highlights (Notebook)

The `Analyzing_Data.ipynb` notebook covers:

- `df.info()` and `df.describe()` — data types, null counts, distributions
- `ocean_proximity` value counts — class imbalance check
- Correlation matrix — `median_income` is the strongest predictor of house value
- Histogram plots for all features
- Stratified income category distribution
- Manual preprocessing experiments (imputation, scaling, encoding)
- StandardScaler vs MinMaxScaler comparison

---

##  Tech Stack

- **Python 3.12**
- **pandas** — data loading and manipulation
- **NumPy** — numerical operations
- **scikit-learn** — pipeline, preprocessing, model, evaluation
- **joblib** — model serialization
- **matplotlib** — visualizations (notebook)
- **Jupyter Notebook** — EDA environment

---

##  Possible Improvements

- Hyperparameter tuning with `GridSearchCV` or `RandomizedSearchCV`
- Feature engineering (e.g., `rooms_per_household`, `bedrooms_per_room`)
- Try other models: `GradientBoostingRegressor`, `XGBoost`, `LightGBM`
- Add RMSE and R² alongside MAPE for more complete evaluation
- Build a simple prediction API using Flask or FastAPI

---

##  Reference

- Dataset: [StatLib — California Housing](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html)
- Inspired by: *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* by Aurélien Géron
