#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os
import pandas as pd
import numpy as np
from functools import reduce
from typing import Union
from datetime import datetime

def combine_inform_data(iso_country_code: str, end_date: Union[str, datetime]) -> None:
    """
    Combine various data sources and save it to a CSV file.

    Parameters:
    - iso_country_code (str): ISO country code.
    - end_date (Union[str, datetime]): End date for the report.

    Returns:
    None
    """
    # Initialize source directory
    src_path = os.path.join(os.path.dirname(os.getcwd()), 'processed/INFORM', iso_country_code)

    # Read and preprocess the data
    combined_df = pd.DataFrame()
    for file_name in ['YTD_Dengue.csv', 'Remote_Sensing.csv', 'POI.csv', 'Weather.csv', 'Demographics.csv']:
        df = pd.read_csv(os.path.join(src_path, file_name))
        if 'Location' in df.columns:
            df["Location"] = df["Location"].apply(lambda l: l.upper().strip())
        else:
            df["Location"] = df["loc"].apply(lambda l: l.upper().strip())
            
        if len(combined_df) == 0:
            combined_df = df
        else:
            combined_df = combined_df.merge(df, on=["Location"], how="left")

    # Data cleaning
    n_cols = combined_df.shape[1]
    thresh_val_cols = n_cols // 3
    thresh_val_rows = 0.1 * len(combined_df)
    combined_df = combined_df.dropna(thresh=thresh_val_cols + 1)
    combined_df = combined_df.dropna(axis=1, thresh=thresh_val_rows)
    combined_df = combined_df.dropna(axis='columns', how='all')

    cols_incl = [c for c in combined_df.columns.tolist() if c in ['school', 'doctors', 'clinic', 'hospital', 'college', 'university',
       'community_centre', 'pharmacy', 'childcare', 'social_facility',
       'Location', 'health_post', 'toilets', 'kindergarten',
       'waste_transfer_station', 'nursing_home', 'prep_school',
       'waste_disposal', 'social_centre', 'healthcare', 'birthing_centre',
       'dormitory', 'quarantine_facility', 'rescue-station', 'house',
       'residential', 'apartments', 'river', 'pond', 'lake', 'lagoon', 'basin',
       'oxbow', 'reservoir', 'canal', 'stream', 'drain', 'wastewater', 'lock',
       'ditch', 'stream_pool', 'moat', 'reflecting_pool']]
    combined_df[cols_incl] = combined_df[cols_incl].fillna(0)
    combined_df = combined_df.dropna()
    
    # Add report date
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    combined_df["report_date"] = end_date

    # Save the processed data
    out_path = os.path.join(os.path.dirname(os.getcwd()), 'processed/INFORM', iso_country_code)
    os.makedirs(out_path, exist_ok=True)
    combined_df.to_csv(os.path.join(out_path, "INFORM.csv"), index=False)

# Example Usage    
# if __name__ == "__main__":
#    combine_inform_data("PHL", "29/12/2019")

