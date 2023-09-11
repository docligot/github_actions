#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os
from datetime import datetime

def process_fcast_data(iso_country_code, adm, flnm):
    """
    Process dengue data along with Google Trends and Meteorological data.

    Parameters:
    - iso_country_code (str): ISO country code
    - adm (str): Administrative region
    - flnm (str): Filename of the dengue data. The file should be saved in the 'Dengue' subfolder in 'data'.
                 It should have the following columns: [loc = subnational area name (all caps), 
                 cases = number of dengue cases, deaths = number of dengue deaths, Region = regional area name (all caps)]

    Returns:
    - None: Saves the processed data to an Excel file
    """

    # Administrative Boundaries Mapping
    admin_file = os.path.join(os.getcwd(), '..', 'data', f"Admin_Boundaries_{iso_country_code}.csv")
    admin_df = pd.read_csv(admin_file)
    region = admin_df[admin_df["Subregion"] == adm]["Region"].tolist()[0]

    # Google Trends Data
    gt_file = os.path.join(os.getcwd(), '..', 'data', 'Google Trends', f"{iso_country_code}.csv")
    gt_df = pd.read_csv(gt_file)
    gt_df = gt_df[gt_df["region"] == region]
    
    # Determine the date format
    sample_date = gt_df["date"].iloc[0]
    if "-" in sample_date:
        date_format = '%Y-%m-%d'
    elif "/" in sample_date:
        date_format = '%d/%m/%Y'
    else:
        print("Unknown date format in Google Trends data.")
        return
    
    gt_df["date"] = gt_df["date"].apply(lambda d: datetime.strptime(d, '%Y-%m-%d'))
    gt_df = gt_df.pivot(index=["date"], columns="keyword", values="value").reset_index().sort_values(by=["date"], ascending=True)

    # Meteorological Data
    weather_file = os.path.join(os.getcwd(), '..', 'data', 'NASA POWER', f"{iso_country_code}.csv")
    weather_df = pd.read_csv(weather_file)
    weather_df = weather_df[weather_df["loc"] == adm.upper()]
    
    # Check if DataFrame is empty
    if weather_df.empty:
        print(f"No weather data found for {adm.upper()}. Skipping...")
        return

    # Check if required columns exist
    if 'YEAR' not in weather_df.columns or 'DOY' not in weather_df.columns:
        print("Required columns 'YEAR' and 'DOY' not found in weather data. Skipping...")
        return

    # If everything is fine, proceed with the date conversion
    weather_df['date'] = weather_df.apply(lambda row: datetime.strptime(f"{int(row['YEAR'])} {int(row['DOY'])}", '%Y %j').strftime('%d/%m/%Y'), axis=1)
    weather_df["date"] = weather_df["date"].apply(lambda d: datetime.strptime(d, '%d/%m/%Y'))
    weather_df = weather_df.drop(columns=["YEAR","DOY"])

    # Combine Google Trends and Meteorological Data for the specific location
    combin_df = gt_df.merge(weather_df, on="date", how="left")

    # Dengue Data
    dengue_path = os.path.join(os.getcwd(), '..', 'data/Dengue', f"{iso_country_code}/", flnm)
    file_name, file_extension = os.path.splitext(dengue_path)

    if file_extension == ".xlsx":
        dengue_df = pd.read_excel(dengue_path)
    elif file_extension == ".csv":
        dengue_df = pd.read_csv(dengue_path)
    else:
        print("Wrong file type. Read either a CSV or an Excel file")
        return

    dengue_df = dengue_df[dengue_df["loc"] == adm.upper()]
    dengue_df = dengue_df.merge(combin_df, on=["loc","date"], how="left")

    # Save the processed data
    data_folder = os.path.join(os.getcwd(), '..', 'processed', 'Forecasting', f"{iso_country_code}")
    os.makedirs(data_folder, exist_ok=True)
    flnm2 = os.path.join(data_folder, f"{adm}.xlsx")
    dengue_df.to_excel(flnm2, index=False)
    print(f"Finished with {adm}")

# Example usage
#iso_country_code = "PHL"
#admin_file = os.path.join(os.getcwd(), '..', 'data', f"Admin_Boundaries_{iso_country_code}.csv")
#admin_df = pd.read_csv(admin_file)
#adm_list = admin_df["Subregion"].tolist()
#for a in adm_list:
#    process_fcast_data(iso_country_code, a, f"{iso_country_code}.xlsx")

