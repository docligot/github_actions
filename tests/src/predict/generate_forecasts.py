#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
import joblib
import pandas as pd
from scalecast.Forecaster import Forecaster

def forecast_data(iso_country_code, adm, target, n_ar, freq):
    parent_folder = os.path.dirname(os.getcwd())
    model_folder = os.path.join(parent_folder, 'model', 'Forecasting', f"{iso_country_code}/{adm}")

    best_model = pd.read_pickle(os.path.join(model_folder, "auto-ts_summary.pickle")).sort_values(by=["TestSetRMSE"], ascending=True).iloc[0]["ModelNickname"]
    print(f"Best Model is: {best_model}")

    best_params = pd.read_pickle(os.path.join(model_folder, "auto-ts_summary.pickle")).sort_values(by=["TestSetRMSE"], ascending=True).iloc[0]["HyperParams"]
    print(f"Hyperparameters of Best Model: {best_params}")
    
    print("Preparing latest data...")

    f1 = joblib.load(os.path.join(model_folder, "forecaster.pkl"))
    t = joblib.load(os.path.join(model_folder, "transformer.pkl"))
    r = joblib.load(os.path.join(model_folder, "reverter.pkl"))

    fcst_length = f1.test_length

    # Data Preparation
    data_folder = os.path.join(parent_folder, 'processed', 'Forecasting', iso_country_code)
    os.makedirs(data_folder, exist_ok=True)
    flnm = os.path.join(data_folder, f"{adm}.xlsx")
    df = pd.read_excel(flnm, parse_dates=True)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    df2 = df.drop(columns=["loc", "deaths", "Region"]) if target == "cases" else df.drop(columns=["loc", "cases", "Region"])

    # Initialize Forecaster for new data
    f = Forecaster(y=df2[target], current_dates=df2["date"])
    f.generate_future_dates(fcst_length)

    # Add exogenous variables to the forecaster
    f.add_ar_terms(n_ar)
    f.add_time_trend()
    f.add_seasonal_regressors('month', 'quarter', raw=False, dummy=True, drop_first=True)
    f.add_covid19_regressor()

    # Include lagged values of meteorological and Google Trends variables as exogenous variables
    addl_Xvars = df2[[c for c in df2.columns.tolist() if c in list(set([c.split("lag")[0][:-1] for c in f1.Xvars])) + ["date"]]]
    last_date = addl_Xvars['date'].iloc[-1]
    new_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=fcst_length, freq=freq)
    new_date_df = pd.DataFrame(new_dates, columns=['date'])
    addl_Xvars = pd.concat([addl_Xvars, new_date_df], ignore_index=True)
    
    for var in [c for c in addl_Xvars.columns if c != "date"]:
        for n in range(fcst_length, fcst_length*2+1):
            addl_Xvars[f'{var}_lag_{n}'] = addl_Xvars[var].shift(n)
    
    cols_to_keep = [col for col in addl_Xvars.columns if col in f1.Xvars]
    addl_Xvars = addl_Xvars[cols_to_keep + ["date"]]
    addl_Xvars = addl_Xvars.fillna(0)
    f.ingest_Xvars_df(addl_Xvars, date_col="date", use_future_dates=True)
    
    # Update forecaster to retain only the regressors used by the best model
    Xvars_to_drop = [X for X in f.get_regressor_names() if X not in f1.Xvars]
    for X in Xvars_to_drop:
        f.drop_Xvars(X)
        
    print("Scoring data...")
    
    # Apply transformation to series
    f = t.fit_transform(f)
    
    f.set_estimator(best_model)
    f.manual_forecast(**best_params)
    f = r.fit_transform(f)
    
    fcast_df = pd.DataFrame(f.y)
    fcast_df.index = df2.date
    col_nm = target.title() + " (Forecast)"
    fcast_df2 = f.export("lvl_fcsts").rename(columns={"DATE": "date", best_model: col_nm}).set_index("date")
    fcast_df2[col_nm] = fcast_df2[col_nm].apply(lambda c: 0 if c < 0 else round(c,0))
    fcast_df = fcast_df.append(fcast_df2)
    fcast_df["Subregion"] = adm
    fcast_df["ISO Country Code"] = iso_country_code
    fcast_df["Date Forecast"] = df2.date.max().strftime('%Y-%m-%d')
    
    print(f"Saving forecasts for {adm}...")
    # Save Models
    parent_folder = os.path.dirname(os.getcwd())
    model_folder = os.path.join(parent_folder, 'processed', 'Dashboard/Forecast', f"{iso_country_code}")
    os.makedirs(model_folder, exist_ok=True)
    fcast_df.to_csv(os.path.join(model_folder, f"{adm}.csv"))
    
    print("Finished")

# Example usage
#forecast_data("PHL", "Zamboanga del Norte", "cases", 12, "W")

