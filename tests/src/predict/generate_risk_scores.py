#!/usr/bin/env python
# coding: utf-8

# In[10]:


from pathlib import Path
import os
import pandas as pd
import joblib
from functools import reduce
from sklearn.preprocessing import StandardScaler
import numpy as np
from datetime import datetime
from functools import reduce

def load_and_prepare_new_data(risk_dimension, ISO_COUNTRY_CODE):
    DATA_DIR = Path(os.path.dirname(os.getcwd()), "data", "INFORM_DF_Variables.csv")
    SRC_DIR = Path(os.path.dirname(os.getcwd()), "processed", "INFORM", ISO_COUNTRY_CODE)
    
    risk_vars = pd.read_csv(DATA_DIR)
    risk_df = pd.read_csv(SRC_DIR / "INFORM.csv")
    selected_var = risk_vars[risk_vars["Risk Dimension"] == risk_dimension].Variable.tolist()
    new_data = risk_df[[col for col in risk_df.columns.tolist() if col in selected_var + ["Location"]]].set_index("Location")
    return new_data

def standardize_new_data(new_data, scaler):
    return pd.DataFrame(scaler.fit_transform(new_data), columns=new_data.columns, index=new_data.index)

def rescale_to_range(data, min_val=1, max_val=10):
    """Rescale the data to fall within the specified range."""
    data_range = np.max(data) - np.min(data)
    if data_range == 0:
        return np.full_like(data, (max_val + min_val) / 2)  # Return the midpoint if range is zero
    else:
        return min_val + (data - np.min(data)) * (max_val - min_val) / data_range

def prepare_new_data(new_data, scaler):
    new_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    new_data.fillna(new_data.mean(), inplace=True)
    new_data_standardized = standardize_new_data(new_data, scaler)
    return new_data_standardized

def load_models(ISO_COUNTRY_CODE, risk_dimension):
    base_path = os.path.join(os.path.dirname(os.getcwd()), "model/INFORM", ISO_COUNTRY_CODE)
    version_dates = [folder for folder in os.listdir(base_path) if folder.isdigit()]
    latest_version_date = max(version_dates)
    latest_folder_path = os.path.join(base_path, latest_version_date)
    
      
    try:
        spca_model = joblib.load(f"{latest_folder_path}/{risk_dimension}_spca_best_model.pkl")
    except FileNotFoundError:
        spca_model = None
    
    lr_model = joblib.load(f"{latest_folder_path}/{risk_dimension}_final_linear_regression_model.pkl")
    
    spc_flags = pd.read_csv(f"{latest_folder_path}/{risk_dimension}_spc_flags.csv", header=None).values.flatten()
    
    return spca_model, lr_model, spc_flags


def score_new_data(ISO_COUNTRY_CODE, risk_dimension, new_data):
    spca_model, lr_model, spc_flags = load_models(ISO_COUNTRY_CODE, risk_dimension)
    
    # Load spc_flags
    #spc_flags_path = os.path.join(os.path.dirname(os.getcwd()), "model/INFORM", f"{ISO_COUNTRY_CODE}/{risk_dimension}_spc_flags.csv")
    #spc_flags = pd.read_csv(spc_flags_path)['spc_flags'].tolist()
    
    # Check if spca_model is available
    if spca_model is not None and spca_model.n_components_ > 1:
        # Assuming new_data is already standardized
        transformed_data = spca_model.transform(new_data)
        
        # Reshape spc_flags for broadcasting
        spc_flags_reshaped = np.reshape(spc_flags, (1, -1))
        
        # Apply spc_flags to ensure the same transformations as during training
        transformed_data_filtered = np.where(spc_flags_reshaped, transformed_data, -transformed_data)
        
        # Rescale the new SPCs
        transformed_data_rescaled = np.apply_along_axis(rescale_to_range, 0, transformed_data_filtered)
        
        Y_pred = lr_model.predict(transformed_data_rescaled)
    else:
        # Use the linear regression model directly on the standardized features
        Y_pred = lr_model.predict(new_data)
        
    return Y_pred, transformed_data_rescaled if 'transformed_data_rescaled' in locals() else None


def score_final_risk(ISO_COUNTRY_CODE, new_data_hazard, new_data_vulnerability, new_data_lack_of_coping_capacity):
    final_model = load_final_model(ISO_COUNTRY_CODE)
    new_data = pd.DataFrame({
        'Hazard': new_data_hazard,
        'Vulnerability': new_data_vulnerability,
        'Lack of Coping Capacity': new_data_lack_of_coping_capacity
    })
    final_Y_pred = final_model.predict(new_data)
    return final_Y_pred



# In[11]:


def load_final_model(ISO_COUNTRY_CODE):
    base_path = os.path.join(os.path.dirname(os.getcwd()), "model/INFORM", f"{ISO_COUNTRY_CODE}")
    version_dates = [folder for folder in os.listdir(base_path) if folder.isdigit()]
    latest_version_date = max(version_dates)
    latest_folder_path = os.path.join(base_path, latest_version_date)
    
    final_model = joblib.load(f"{latest_folder_path}/INFORM_final_linear_regression_model.pkl")
    
    return final_model

def risk_score_data(ISO_COUNTRY_CODE):
    risk_dimensions = ["Hazard", "Vulnerability", "Lack of Coping Capacity"]
    scaler_X = StandardScaler()  # Replace with the actual scaler used during training
    dfs = []
    
    for risk_dimension in risk_dimensions:
        new_data_raw = load_and_prepare_new_data(risk_dimension, ISO_COUNTRY_CODE)
        new_data = prepare_new_data(new_data_raw, scaler_X)
        
        Y_pred, _ = score_new_data(ISO_COUNTRY_CODE, risk_dimension, new_data)
        Y_pred_df = pd.DataFrame(Y_pred, columns=[f"{risk_dimension}_Score"])
        
        # Handling SPC scores
        exploded_df = pd.DataFrame(_, columns=[f"{risk_dimension}_SPC_{i+1}" for i in range(_.shape[1])])
        Y_pred_df = pd.concat([Y_pred_df, exploded_df], axis=1)
        
        dfs.append(Y_pred_df)
    
    # Merge the dataframes
    consolidated_df = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True), dfs)
    
    # Temporarily drop the SPC columns
    spc_columns = [col for col in consolidated_df.columns if 'SPC' in col]
    consolidated_df_temp = consolidated_df.drop(columns=spc_columns)
    consolidated_df_temp.columns = [c.replace("Score","Y_pred") for c in consolidated_df_temp.columns.tolist()]
    
    # Load the final model and compute the final Risk score
    final_model = load_final_model(ISO_COUNTRY_CODE)
    consolidated_df['Final_Risk_Score'] = final_model.predict(consolidated_df_temp)

    consolidated_df.index = new_data.index
    
    # Include population-at-risk per location
    popn_df = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()), "processed/INFORM", f"{ISO_COUNTRY_CODE}/INFORM.csv")).set_index("Location")
    popn_df = popn_df[[c for c in popn_df.columns if "population" in c]]
    consolidated_df = consolidated_df.merge(popn_df, left_index=True, right_index=True, how="left")
    
    # Include reporting_date
    consolidated_df['reporting_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    
    # Export to CSV
    out_dir = os.path.join(os.path.dirname(os.getcwd()), "processed/Dashboard/INFORM", ISO_COUNTRY_CODE)
    os.makedirs(out_dir, exist_ok=True)
    consolidated_df.to_csv(os.path.join(out_dir, "INFORM_scores.csv"))

# Example Usage
risk_score_data(ISO_COUNTRY_CODE)

