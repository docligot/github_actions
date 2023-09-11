#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import SparsePCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from bayes_opt import BayesianOptimization
from scipy.special import expit
import os
from datetime import datetime
from functools import reduce


# In[9]:


def load_and_prepare_data(risk_dimension, ISO_COUNTRY_CODE, TARGET_VAR):
    DATA_DIR = Path(os.path.dirname(os.getcwd()), "data", "INFORM_DF_Variables.csv")
    SRC_DIR = Path(os.path.dirname(os.getcwd()), "processed", "INFORM", ISO_COUNTRY_CODE)
    
    risk_vars = pd.read_csv(DATA_DIR)
    risk_df = pd.read_csv(SRC_DIR / "INFORM.csv")
    selected_var = risk_vars[risk_vars["Risk Dimension"] == risk_dimension].Variable.tolist()
    X = risk_df[[col for col in risk_df.columns.tolist() if col in selected_var + ["Location"]]].set_index("Location")
    Y = risk_df[TARGET_VAR]
    return X, Y


def rescale_to_range(data, min_val=1, max_val=10):
    """Rescale the data to fall within the specified range."""
    data_range = np.max(data) - np.min(data)
    if data_range == 0:
        return np.full_like(data, (max_val + min_val) / 2)  # Return the midpoint if range is zero
    else:
        return min_val + (data - np.min(data)) * (max_val - min_val) / data_range

def standardize_data(X, Y):
    """
    Standardizes the feature matrix X and rescales the target variable Y.
    
    Parameters:
    - X: DataFrame containing the feature matrix.
    - Y: Series containing the target variable.
    
    Returns:
    - X_standardized: DataFrame containing the standardized feature matrix.
    - Y_rescaled: Series containing the rescaled target variable.
    """
    
    # Check for NaN or inf in the Data
    if np.any(np.isnan(X)) or np.any(np.isnan(Y)):
        print("Warning: NaN values detected in the data.")
    if np.any(np.isinf(X)) or np.any(np.isinf(Y)):
        print("Warning: Infinite values detected in the data.")
    
    # Standardize the features
    scaler_X = StandardScaler()
    X_standardized = pd.DataFrame(scaler_X.fit_transform(X), columns=X.columns, index=X.index)
    
    # Rescale the target variable using custom function
    Y_rescaled = pd.Series(rescale_to_range(Y), index=Y.index)
    
    return X_standardized, Y_rescaled


def refine_spcs(X_spcs, X_standardized, Y_standardized):
    """Refine the SPCs based on the provided criteria."""
    # Initialize a list to store boolean flags
    spc_flags = []
    
    def ensure_positive_correlation(data, target):
        nonlocal spc_flags  # Declare spc_flags as nonlocal
        correlation = np.corrcoef(data, target)[0, 1]
        if correlation >= 0:
            spc_flags.append(True)
            return data
        else:
            spc_flags.append(False)
            return -data
    
    # First, ensure positive correlation
    X_spcs_positive_corr = np.apply_along_axis(ensure_positive_correlation, 0, X_spcs, Y_standardized)
    
    # Next, rescale
    X_spcs_refined = np.apply_along_axis(rescale_to_range, 0, X_spcs_positive_corr)
    
    # Convert the numpy array to a DataFrame and ensure the index matches the original data
    original_index = X_standardized.index
    refined_df = pd.DataFrame(X_spcs_refined, columns=[f"SPC_{i+1}" for i in range(X_spcs_refined.shape[1])], index=original_index)
    
    return refined_df, spc_flags

def spca_objective(n_components, alpha, ridge_alpha, beta, gamma, X_standardized):
    n_components = int(n_components)
    spca = SparsePCA(n_components=n_components, alpha=alpha, ridge_alpha=ridge_alpha)
    
    # Set up k-fold cross-validation
    kf = KFold(n_splits=5)  # 5-fold cross-validation
    explained_variance_ratios = []
    num_non_zero_loadings_list = []
    
    # Perform k-fold cross-validation
    for train_index, test_index in kf.split(X_standardized):
        X_train, X_test = X_standardized.iloc[train_index], X_standardized.iloc[test_index]
        
        # Fit the model on the training data
        transformed_data = spca.fit_transform(X_train)
        
        # Compute the explained variance for each component
        explained_variance = np.var(transformed_data, axis=0)
        
        # Compute the total variance of the original data
        total_variance = np.var(X_train, axis=0).sum()
        
        # Compute the explained variance ratio
        explained_variance_ratio = explained_variance.sum() / total_variance
        
        # Append the explained variance ratio for this fold
        explained_variance_ratios.append(explained_variance_ratio)
        
        # Calculate the number of non-zero loadings for this fold
        num_non_zero_loadings = np.count_nonzero(spca.components_)
        num_non_zero_loadings_list.append(num_non_zero_loadings)
    
    # Compute the average explained variance ratio across all folds
    avg_explained_variance_ratio = np.mean(explained_variance_ratios)
    
    # Compute the average number of non-zero loadings
    avg_non_zero_loadings = np.mean(num_non_zero_loadings_list)
    
    # Calculate the penalty term
    complexity_term = n_components + alpha + ridge_alpha
    penalty = expit(beta * (avg_non_zero_loadings - gamma + complexity_term))
    
    # Calculate the adjusted explained variance ratio
    adjusted_explained_variance_ratio = avg_explained_variance_ratio * (1 - penalty)
    
    return adjusted_explained_variance_ratio

def run_spca_optimization(X_standardized):
    # Update the bounds dictionary
    bounds = {
        'n_components': (1, round(X_standardized.shape[0] / 20, 0)),
        'alpha': (0.01, 10),
        'ridge_alpha': (0.01, 10),
        'beta': (0.1, 2),  # You can set the range based on your specific needs
        'gamma': (0, 50)  # You can set the range based on your specific needs
        }
    
    # Initialize the Bayesian optimizer
    optimizer = BayesianOptimization(
        f=lambda n_components, alpha, ridge_alpha, beta, gamma: spca_objective(n_components, alpha, ridge_alpha, beta, gamma, X_standardized),
        pbounds=bounds,
        random_state=42,
        allow_duplicate_points=True
    )

    # Run the optimizer
    optimizer.maximize(init_points=5, n_iter=200)
    
    # Extracting the best parameters from the Bayesian optimization of SPCA
    best_params = optimizer.max['params']
    best_params['n_components'] = int(best_params['n_components'])
    
    return best_params



def rfe_objective(n_features_to_select, X_for_LR, Y_rescaled):
    n_features_to_select = int(n_features_to_select)
    
    # Initialize Linear Regression Model
    model = LinearRegression()
    
    # Initialize RFE Selector
    selector = RFE(estimator=model, n_features_to_select=n_features_to_select)
    
    # Fit RFE Selector using the data prepared for RFE
    selector = selector.fit(X_for_LR, Y_rescaled)
    
    # Cross-validated MSE (using 5-fold CV as an example)
    scores = cross_val_score(selector, X_for_LR, Y_rescaled, cv=5, scoring='neg_mean_squared_error')
    
    # Return the average negative MSE
    return scores.mean()

def run_rfe_optimization(X_for_LR, Y_rescaled):
    # RFE optimization bounds
    bounds_rfe = {
        'n_features_to_select': (1, X_for_LR.shape[1])
    }
    
    # Initialize Bayesian Optimization for RFE
    optimizer_rfe = BayesianOptimization(
        f=lambda n_features_to_select: rfe_objective(n_features_to_select, X_for_LR, Y_rescaled),
        pbounds=bounds_rfe,
        random_state=42,
        allow_duplicate_points=True
    )
    
    # Run the Bayesian Optimization for RFE
    optimizer_rfe.maximize(init_points=5, n_iter=200)
    
    # Extract the best number of features to select
    best_n_features_to_select = int(optimizer_rfe.max['params']['n_features_to_select'])
    
    return best_n_features_to_select

# Function to save models and metrics
#def save_models_and_metrics(ISO_COUNTRY_CODE, spca_model, lr_model, selector_model, optimizer, mean_cv_score, std_cv_score):
def save_models_and_metrics(ISO_COUNTRY_CODE, risk_dimension, spca_model, spc_flags, lr_model, X, Y, Y_pred, mean_cv_score, std_cv_score):
    base_path = os.path.join(os.path.dirname(os.getcwd()), "model/INFORM", f"{ISO_COUNTRY_CODE}")
    
    # Get current date in yyyymmdd format
    version_date = datetime.now().strftime("%Y%m%d")
    
    # Create subfolder named as the version date
    save_path = os.path.join(base_path, version_date)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Save models using joblib
    if spca_model is not None:
        joblib.dump(spca_model, f"{save_path}/{risk_dimension}_spca_best_model.pkl")
    if lr_model is not None:
        joblib.dump(lr_model, f"{save_path}/{risk_dimension}_final_linear_regression_model.pkl")
    #if selector_model is not None:
    #    joblib.dump(selector_model, f"{save_path}/{risk_dimension}_final_linear_regression_rfe_selector.pkl")
    #if optimizer is not None:
    #    joblib.dump(optimizer, f"{save_path}/{risk_dimension}_spca_optimizer.pkl")
    if spca_model.n_components_ > 1:
        loadings_df = pd.DataFrame(spca_model.components_, columns=spca_model.feature_names_in_).T
        loadings_df.columns = [f"SPC_{i+1}" for i in range(spca_model.n_components_)]
    if loadings_df is not None:
        loadings_df.to_pickle(f"{save_path}/{risk_dimension}_spca_loadings.pickle")
    if X is not None:
        X.to_csv(f"{save_path}/{risk_dimension}_X.csv")
    if Y is not None:
        Y.to_csv(f"{save_path}/{risk_dimension}_Y.csv")
    if Y_pred is not None:
        Y_pred.to_csv(f"{save_path}/{risk_dimension}_Y_pred.csv")
    if spc_flags is not None:
        # Save the boolean flags for SPCs
        with open(f"{save_path}/{risk_dimension}_spc_flags.csv", "w") as f:
            for flag in spc_flags:
                f.write(f"{flag}\n")


    # Save metrics using pandas
    metrics_df = pd.DataFrame({
        'Mean_CV_Score': [mean_cv_score],
        'Std_CV_Score': [std_cv_score]
    })
    metrics_df.to_csv(f"{save_path}/{risk_dimension}_model_metrics.csv", index=False)
                            


# In[10]:


# Main code
def train_risk_model(ISO_COUNTRY_CODE, TARGET_VAR, risk_dimension):
    X, Y = load_and_prepare_data(risk_dimension, ISO_COUNTRY_CODE, TARGET_VAR)
    X_standardized, Y_standardized = standardize_data(X, Y)
    best_spca_params = run_spca_optimization(X_standardized)
    n_components_best = best_spca_params['n_components']
    cv_scores = None
    spca_best = None
    lr = None
    Y_pred = None
    spc_flags = None

    if n_components_best == 1:
        best_n_features_to_select = run_rfe_optimization(X_standardized, Y_standardized)
        lr = LinearRegression()
        selector = RFE(estimator=lr, n_features_to_select=best_n_features_to_select)
        selector.fit(X_standardized, Y_standardized)
        # Generate predicted Y values
        Y_pred = pd.DataFrame(selector.predict(X_standardized), index=X_standardized.index, columns=[f"{risk_dimension}_Y_pred"])
        cv_scores = cross_val_score(selector, X_standardized, Y_standardized, cv=5, scoring='neg_mean_squared_error')
    else:
        spca_best = SparsePCA(n_components=n_components_best, alpha=best_spca_params['alpha'], ridge_alpha=best_spca_params['ridge_alpha'])
        X_spcs = spca_best.fit_transform(X_standardized)
        X_for_LR, spc_flags = refine_spcs(X_spcs, X_standardized, Y_standardized)
        lr = LinearRegression()
        lr.fit(X_for_LR, Y_standardized)
        # Generate predicted Y values
        Y_pred = pd.DataFrame(lr.predict(X_for_LR), index=X_standardized.index, columns=[f"{risk_dimension}_Y_pred"])
        cv_scores = cross_val_score(lr, X_for_LR, Y_standardized, cv=5, scoring='neg_mean_squared_error')

    if cv_scores is not None:
        mean_cv_score = np.mean(cv_scores)
        std_cv_score = np.std(cv_scores)

    save_models_and_metrics(ISO_COUNTRY_CODE, risk_dimension, spca_best, spc_flags, lr, X, Y, Y_pred, mean_cv_score, std_cv_score)


# In[15]:


def train_final_risk_model(ISO_COUNTRY_CODE, TARGET_VAR):
    SRC_DIR = Path(os.path.dirname(os.getcwd()), "processed", "INFORM", ISO_COUNTRY_CODE)
    risk_df = pd.read_csv(SRC_DIR / "INFORM.csv").set_index("Location")
    Y = risk_df[TARGET_VAR]
    Y_rescaled = rescale_to_range(Y)
        
    # Step 1: Find the latest version date folder
    base_path = os.path.join(os.path.dirname(os.getcwd()), "model/INFORM", f"{ISO_COUNTRY_CODE}")
    version_dates = [folder for folder in os.listdir(base_path) if folder.isdigit()]
    latest_version_date = max(version_dates)
    
    # Step 2: Check for the availability of Y_pred files
    latest_folder_path = os.path.join(base_path, latest_version_date)
    required_files = ["Hazard_Y_pred.csv", "Vulnerability_Y_pred.csv", "Lack of Coping Capacity_Y_pred.csv"]
    
    if all(os.path.exists(os.path.join(latest_folder_path, file)) for file in required_files):
        # Step 3: Load the Y_pred files and prepare the feature matrix for the final model
        hazard_y_pred = pd.read_csv(os.path.join(latest_folder_path, "Hazard_Y_pred.csv"))
        vulnerability_y_pred = pd.read_csv(os.path.join(latest_folder_path, "Vulnerability_Y_pred.csv"))
        lack_of_coping_capacity_y_pred = pd.read_csv(os.path.join(latest_folder_path, "Lack of Coping Capacity_Y_pred.csv"))
        
        # Concatenate the Y_pred values to form the feature matrix
        X_final = reduce(lambda  left,right: pd.merge(left,right,on=['Location'], how='left'), [hazard_y_pred, vulnerability_y_pred, lack_of_coping_capacity_y_pred])
        X_final = X_final.set_index("Location")
        
        # Step 4: Run linear regression and perform 5-fold CV
        final_model = LinearRegression()
        final_model.fit(X_final, Y_rescaled.values)
        
        # Perform 5-fold cross-validation
        cv_scores = cross_val_score(final_model, X_final, Y_rescaled, cv=5, scoring='neg_mean_squared_error')
        
        mean_cv_score = cv_scores.mean()
        std_cv_score = cv_scores.std()
        
        print(f"5-Fold CV Mean Score: {mean_cv_score}")
        print(f"5-Fold CV Standard Deviation: {std_cv_score}")
        
        # Save predicted Y
        Y_pred = pd.DataFrame(final_model.predict(X_final), index=Y_rescaled.index, columns=["INFORM_Y_pred"])
    else:
        print("Required Y_pred files are missing.")

    
    # Save models using joblib
    if final_model is not None:
        joblib.dump(final_model, f"{latest_folder_path}/INFORM_final_linear_regression_model.pkl")
    if Y_pred is not None:
        Y_pred.to_csv(f"{latest_folder_path}/INFORM_Y_pred.csv")

    # Save metrics using pandas
    metrics_df = pd.DataFrame({
        'Mean_CV_Score': [mean_cv_score],
        'Std_CV_Score': [std_cv_score]
    })
    metrics_df.to_csv(f"{latest_folder_path}/INFORM_model_metrics.csv", index=False)
    print("INFORM risk model development completed.")
    
# Example usage
#train_risk_model("PHL", "CFR", "Hazard")
#train_risk_model("PHL", "CFR", "Vulnerability")
#train_risk_model("PHL", "CFR", "Lack of Coping Capacity")
#train_final_risk_model("PHL", "CFR")

