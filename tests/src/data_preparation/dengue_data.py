#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os
import pandas as pd
import numpy as np
from typing import Union
from datetime import datetime

def process_dengue_data(iso_country_code: str, adm_level: int, end_date: Union[str, datetime]) -> None:
    """
    Process Dengue data and save it to a CSV file.

    Parameters:
    - iso_country_code (str): ISO country code.
    - adm_level (int): Administrative level.
    - end_date (Union[str, datetime]): End date for filtering data.

    Returns:
    None
    """
    # Initialize source directory and file paths
    src_dir = os.path.join(os.path.dirname(os.getcwd()), 'data/Dengue', iso_country_code)
    file_name = os.listdir(src_dir)[0]
    file_name, file_extension = os.path.splitext(file_name)

    # Read the data
    if file_extension == ".xlsx":
        dengue_df = pd.read_excel(os.path.join(src_dir, f"{file_name}{file_extension}"))
    elif file_extension == ".csv":
        dengue_df = pd.read_csv(os.path.join(src_dir, f"{file_name}{file_extension}"))
    else:
        print("Wrong file type. Read either a CSV or an Excel file")
        return

    # Convert end_date to datetime if it's a string
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)

    # Compute start date as the first day of the year of the end date
    start_date = pd.to_datetime(f"{end_date.year}-01-01")

    # Filter and process the data
    dengue_df = dengue_df[(dengue_df['date'] >= start_date) & (dengue_df['date'] <= end_date)]
    dengue_df2 = dengue_df.groupby('loc')[['cases', 'deaths']].sum().reset_index()
    dengue_df2["CFR"] = dengue_df2.apply(lambda x: x["deaths"] / x["cases"] if x["cases"] > 0 else 0, axis=1)
    dengue_df2 = dengue_df2.rename(columns={"loc": "Location"})
    dengue_df2["Location"] = dengue_df2["Location"].apply(lambda l: l.upper().strip())
    dengue_df2["report_date"] = end_date

    # Save the processed data
    out_path = os.path.join(os.path.dirname(os.getcwd()), 'processed/INFORM', iso_country_code)
    os.makedirs(out_path, exist_ok=True)
    dengue_df2.to_csv(os.path.join(out_path, "YTD_Dengue.csv"), index=False)

if __name__ == "__main__":
    process_dengue_data("PHL", 2, "2019-12-29")

