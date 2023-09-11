#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os
import joblib
import pandas as pd
import numpy as np
from tsmoothie.smoother import LowessSmoother
from statsmodels.tsa.stattools import grangercausalitytests
from scalecast.Forecaster import Forecaster
from scalecast.Pipeline import Pipeline
from scalecast.util import find_optimal_transformation
from scalecast.multiseries import export_model_summaries
from scalecast import GridGenerator

def outlier_correction (df, var):
    data = df[var].values.reshape(1, -1)

    # operate smoothing
    smoother = LowessSmoother(smooth_fraction=0.25, iterations=1)
    smoother.smooth(data)

    # generate intervals
    low, up = smoother.get_intervals('prediction_interval')

    points = smoother.data[0]
    up_points = up[0]
    low_points = low[0]

    outliers = []

    for i in range(len(points)-1, 0, -1):
        current_point = points[i]
        current_up = up_points[i]
        current_low = low_points[i]
        if current_point > current_up or current_point < current_low:
            outliers.append(i)
            
        
    if len(outliers) > 0:
        for i in outliers:
            df[var].loc[i] = np.NaN
            
        df[var] = df[var].interpolate(option='spline')
        
def grangers_causation_matrix(data, target, maxlag=12, test='ssr_chi2test', verbose=False):    
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    variables = data.select_dtypes([np.int64,np.float64]).columns
    df = pd.DataFrame(np.zeros((1, len(variables))), columns=variables, index=[target])
    for c in df.columns:
        test_result = grangercausalitytests(data[[target, c]], maxlag=maxlag, verbose=False)
        p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
        if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
        min_p_value = np.min(p_values)
        df.loc[target, c] = min_p_value
    df.columns = [var for var in variables]
    df.index = [str(target)]
    return df.columns[df.loc[target].le(0.05)].tolist()

def train_fcast_model(iso_country_code, adm, target, fcst_length, n_ar, freq, n_fold, m):
    """Main function to execute the forecasting pipeline."""
    # Data Preparation
    data_folder = os.path.join(os.path.dirname(os.getcwd()), 'processed', 'Forecasting', iso_country_code)
    os.makedirs(data_folder, exist_ok=True)
    flnm = os.path.join(data_folder, f"{adm}.xlsx")
    df = pd.read_excel(flnm, parse_dates=True)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    df2 = df.drop(columns=["loc", "deaths", "Region"]) if target == "cases" else df.drop(columns=["loc", "cases", "Region"])

    # Outlier Correction
    for v in df2.drop(columns="date").columns:
        outlier_correction(df2, v)

    # Granger Causality Test
    shortlist_var = grangers_causation_matrix(df2, target)

    # Initialize Forecaster
    f = Forecaster(y=df2[target], current_dates=df2["date"])
    f.generate_future_dates(fcst_length)
    f.set_test_length(fcst_length)
    f.set_validation_length(fcst_length)

    # Add exogenous variables to the forecaster
    # f.add_ar_terms(n_ar)
    # f.add_time_trend()
    f.add_seasonal_regressors('month', 'quarter', raw=False, dummy=True, drop_first=True)
    f.add_covid19_regressor()

    #---- Include lagged values of meteorological and Google Trends variables as exogenous variables
    addl_Xvars = df2[shortlist_var + ["date"]]
    last_date = addl_Xvars['date'].iloc[-1]
    new_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=fcst_length, freq=freq)
    new_date_df = pd.DataFrame(new_dates, columns=['date'])
    addl_Xvars = pd.concat([addl_Xvars, new_date_df], ignore_index=True)
    for var in shortlist_var:
        for n in range(fcst_length,fcst_length*2+1):
            addl_Xvars[f'{var}_lag_{n}'] = addl_Xvars[var].shift(n)

    cols_to_keep = [col for col in addl_Xvars.columns if "lag_" in col]
    addl_Xvars = addl_Xvars[cols_to_keep + ["date"]]
    addl_Xvars = addl_Xvars.fillna(0)
    f.ingest_Xvars_df(addl_Xvars,date_col="date",use_future_dates=True)
    
    # Reduces the regressor variables stored in the forecaster
    print("Starting variable reduction...")
    print("Starting out with {} variables".format(len(f.get_regressor_names())))
    f.reduce_Xvars(
        method = 'l1',            # uses a lasso regressor and grid searches for the optimal alpha on the validation
        overwrite = False,
        dynamic_testing = False,  
        dynamic_tuning = fcst_length,
        cross_validate = True,
    )
    lasso_reduced_vars = f.reduced_Xvars[:]
    print(f"Lasso reduced to {len(lasso_reduced_vars)} variables")
    
    # Update forecaster to retain only the lasso reduced regressors
    Xvars_to_drop = [X for X in f.get_regressor_names() if X not in lasso_reduced_vars]
    print("Retaining lasso-reduced exogenous variables...")
    for X in Xvars_to_drop:
        f.drop_Xvars(X)
        
    GridGenerator.get_example_grids()
    
    f_pipe_aut = f
    
     # Hyperparameter tuning setup
    def forecaster_aut(f, models, n_fold, fcst_length):
        f.tune_test_forecast(
            models,
            cross_validate=True,
            feature_importance=True,
            k=n_fold,
            dynamic_tuning=fcst_length,
            dynamic_testing=fcst_length,
        )

    transformer_aut, reverter_aut = find_optimal_transformation(
        f_pipe_aut,
        lags = fcst_length,
        m = m,
        monitor = 'rmse',
        estimator = 'elasticnet',
        test_length = fcst_length,
        num_test_sets = 3,
        space_between_sets = 1,
        verbose = True,
    ) 
    
    pipeline_aut = Pipeline(
        steps = [
            ('Transform',transformer_aut),
            ('Forecast',forecaster_aut),
            ('Revert',reverter_aut),
        ]
    )
    
    print("Starting the model training...")
    f_pipe_aut = pipeline_aut.fit_predict(
        f_pipe_aut,
        models=[
            'mlr',
            'lasso',
            'ridge',
            'elasticnet',
            'xgboost',
            'lightgbm',
            'knn',
            "catboost", 
            "gbt", 
        ], n_fold = n_fold, fcst_length = fcst_length, 
    )
    
    results = f_pipe_aut.export(cis=True,models=['mlr', 'lasso', 'ridge', 'elasticnet', 'xgboost', 'lightgbm', 'knn', "catboost", "gbt"])
    
    model_summary = results['model_summaries'][['ModelNickname','HyperParams','TestSetRMSE','TestSetR2','InSampleRMSE','InSampleR2']]
    model_summary["iso_country_code"] = iso_country_code
    model_summary["adm"] = adm
    model_summary["dateVersion"] = pd.Timestamp.now()
    
    print("Saving models...")
    
    # Save Models
    parent_folder = os.path.dirname(os.getcwd())
    model_folder = os.path.join(parent_folder, 'model', 'Forecasting', f"{iso_country_code}/{adm}")
    os.makedirs(model_folder, exist_ok=True)
    
    best_model = model_summary.sort_values(by="TestSetRMSE", ascending=True)
    best_model.to_pickle(os.path.join(model_folder, "auto-ts_summary.pickle"))
    joblib.dump(f_pipe_aut, os.path.join(model_folder, "forecaster.pkl"))
    joblib.dump(transformer_aut, os.path.join(model_folder, "transformer.pkl"))
    joblib.dump(reverter_aut, os.path.join(model_folder, "reverter.pkl"))

    print("Pipeline run completed.")
    
# Example usage:
# train_fcast_model("PHL", "Zamboanga del Sur", "cases", 4, 12, 'W', 5, 52)

