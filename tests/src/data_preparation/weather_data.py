#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import pandas as pd
import numpy as np
from typing import Union

def process_weather_data(iso_country_code: str, end_date: Union[str, pd.Timestamp]) -> None:
    """
    Process weather data for a given country and time period.

    Parameters:
    - iso_country_code (str): ISO country code
    - end_date (Union[str, pd.Timestamp]): End date in 'YYYY-MM-DD' format or as a pandas Timestamp

    Returns:
    None
    """
    # Construct the file path
    file_path = os.path.join(os.path.dirname(os.getcwd()), 'data/NASA POWER', f"{iso_country_code}.csv")
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Convert 'YEAR' and 'DOY' into a datetime column
    df['Date'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['DOY'].astype(str), format='%Y-%j')

    # Convert end_date to pandas Timestamp if it's a string
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)

    # Compute start date as 3 months before the end date
    start_date = end_date - pd.DateOffset(months=3)

    # Filter rows that fall within the 3-month period
    df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    # Compute mean of each group
    grouped = df_filtered.groupby('loc').mean()

    # Drop unnecessary columns and reset index
    grouped = grouped.drop(columns=["YEAR", "DOY"]).reset_index()

    # Create output directory if it doesn't exist
    out_path = os.path.join(os.path.dirname(os.getcwd()), 'processed/INFORM', f"{iso_country_code}")
    os.makedirs(out_path, exist_ok=True)

    # Save the processed DataFrame to a CSV file
    grouped.to_csv(os.path.join(out_path, "Weather.csv"), index=False)

# Example usage    
#if __name__ == "__main__":
#    iso_country_code = "PHL"  # Replace with your ISO country code
#    end_date = "2019-12-29"  # Replace with your end date in 'YYYY-MM-DD' format
#    process_weather_data(iso_country_code, end_date)


# In[ ]:




