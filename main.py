import os
import joblib

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

MODEL_FILE = 'model.pkl' # model.pkl: will store your trained ML model 
PIPELINE_FILE = 'pipeline.pkl' # we are pickling pipeline bcz we need to transform incoming data (data came for inference) as well, as we did with training data. Bcz incoming data is not scaled, imputed or OneHotEncoded etc. Means we will pass inference data too from a pipeline

def build_pipeline(num_attribs, cat_attribs):
    # For numerical cols
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scalar", StandardScaler())
    ])

    # For categorical cols

    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore")) # if the incoming data contains such values that were not in training, it will ignore those values
        # If we had used OrdinalEncoder instead of OneHotEncoder, we would also use StandardScaler here to scale the encoded categorical columns. OneHotEncoder already gives result in 0s and 1s
    ])

    # Construct the full pipeline

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs), # apply num_pipeline on num_attribs (a list) and the name of pipeline is "num"
        ("cat", cat_pipeline, cat_attribs) # apply cat_pipeline on cat_attribs (a list) and the name of pipeline is "cat"
    ])

    return full_pipeline

if not os.path.exists(MODEL_FILE): # If model file does not exist, we need to train the model
    # Load the DataSet

    housing= pd.read_csv("housing.csv")

    # Create a Stratified test set

    housing['income_cat'] = pd.cut( housing['median_income'], bins=[0, 1.5, 3, 4.5, 6, np.inf], labels=[1, 2, 3, 4, 5])

    split= StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(housing, housing['income_cat']):
        housing.iloc[test_index]["median_house_value"].to_csv("True_labels.csv", index=False) # saving labels in a csv to compare the model predictions with actual values
        housing.iloc[test_index].drop(["income_cat", "median_house_value"], axis=1).to_csv("input.csv", index=False) # Converting the test set into a csv, so that we can make predictions when model is Ready! (Also removing median_house_value col bcz we dont need to pass it to model while making predictions)
        housing = housing.iloc[train_index].drop('income_cat', axis=1) # take training data from housing (df) and drop income_cat column from it

    # Separate features and labels

    housing_labels= housing["median_house_value"].copy()
    housing_features= housing.drop("median_house_value", axis=1)

    # Separate numerical and Categorical cols

    num_attribs = housing_features.drop("ocean_proximity", axis=1).columns.tolist() # It will give col names of housing (df) in the form of list, except ocean_proximity
    cat_attribs = ["ocean_proximity"]

    pipeline= build_pipeline(num_attribs, cat_attribs)

    housing_prepared = pipeline.fit_transform(housing_features)
    # print(housing_prepared)

    # Data has been transformed, Now, Let's train the model

    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared, housing_labels)

    # If u want to add hyperparameter tuning, just add that tuning code here in place of above two lines and keep everything else intact

    joblib.dump(model, MODEL_FILE) # This fn dumps model into MODEL_FILE (Save (serialize) an object to disk)
    joblib.dump(pipeline, PIPELINE_FILE)

    print("Model is trained. Congrats!!!")

else: # If model file exists, then do Inference
    
    model = joblib.load(MODEL_FILE) # Load (deserialize) it back into memory
    pipeline = joblib.load(PIPELINE_FILE)

    input_data= pd.read_csv('input.csv')
    transformed_input = pipeline.transform(input_data) # We are transforming only (not fitting), because the pipeline was already fit on training data and saved earlier

    predictions= model.predict(transformed_input)
    input_data['predicted_median_house_value'] = predictions

    input_data.to_csv("output.csv", index=False)

    print("Inference is Complete. Results are saved into output.csv") 

    true_labels = pd.read_csv("True_labels.csv").values.flatten()  # Convert to 1D array if needed
    mape = np.mean(np.abs((true_labels - predictions) / true_labels)) * 100
    print(f"MAPE (Mean Absolute Percentage Error): {mape:.2f}%")