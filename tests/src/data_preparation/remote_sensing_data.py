#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import numpy as np
import pandas as pd
import rasterio
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)

def read_tif_data(file_path: str) -> np.ndarray:
    """Read GeoTIFF data from a file."""
    with rasterio.open(file_path) as src:
        return src.read()

def filter_data(data: np.ndarray, missing_value: int = -9999) -> np.ndarray:
    """Filter out missing values from the data."""
    return data[data != missing_value]

def get_tif_mean(file_path: str) -> float:
    """Calculate the mean value of a GeoTIFF file."""
    data = read_tif_data(file_path)
    non_missing_data = filter_data(data)
    return np.nan if non_missing_data.size == 0 else non_missing_data.mean()

def get_non_missing_ratio(file_path: str) -> float:
    """Calculate the ratio of non-missing values in a GeoTIFF file."""
    data = read_tif_data(file_path)
    return np.sum(data != -9999) / data.size

def get_closest_tif_mean(index_dir: str, end_date: datetime, non_missing_threshold: float = 0.9) -> (float, datetime):
    """Get the mean value and date of the closest GeoTIFF file to a given end date."""
    try:
        tif_files = [f for f in os.listdir(index_dir) if f.endswith('.tif')]
        tif_files = [f for f in tif_files if get_non_missing_ratio(os.path.join(index_dir, f)) >= non_missing_threshold]
        
        tif_dates = []
        for f in tif_files:
            try:
                tif_dates.append(datetime.strptime(f.split('_')[2][3:], '%Y%j'))
            except ValueError:
                logging.warning(f"Date in {f} does not match format '%Y%j'")
                return np.nan, np.nan
        
        if tif_dates:
            closest_date = min(tif_dates, key=lambda x: abs(x - end_date))
            closest_tif = tif_files[tif_dates.index(closest_date)]
            mean_val = get_tif_mean(os.path.join(index_dir, closest_tif))
        else:
            return np.nan, np.nan
    except Exception as e:
        logging.error(f"Error in {index_dir}: {str(e)}")
        return np.nan, np.nan

    return mean_val, closest_date

def process_remote_sensing_data(iso_country_code: str, end_date_str: str) -> pd.DataFrame:
    """Construct the dataframe."""
    # Construct the parent_folder path using the iso_country_code
    parent_folder = os.path.join(os.path.dirname(os.getcwd()), 'data/NASA AppEEARS', f"{iso_country_code}")
    
    # Parse the date in %d/%m/%Y format
    end_date = datetime.strptime(end_date_str, '%d/%m/%Y')
    
    # List the locations based on the parent_folder
    locations = os.listdir(parent_folder)
    indices = ['NDVI', 'NDWI', 'NDBI']

    data = []
    for loc in locations:
        row = {'Location': loc}
        for idx in indices:
            idx_dir = os.path.join(parent_folder, loc, idx)
            if os.path.exists(idx_dir):
                mean_val, date_val = get_closest_tif_mean(idx_dir, end_date)
                row[idx] = mean_val
                row[idx + '_date'] = date_val
            else:
                row[idx] = np.nan
                row[idx + '_date'] = np.nan
        data.append(row)
    
    rs_df = pd.DataFrame(data)
    rs_df["date_version"] = end_date
    rs_df = rs_df.dropna()
    out_path = os.path.join(os.path.dirname(os.getcwd()), 'processed/INFORM',f'{iso_country_code}')
    os.makedirs(out_path, exist_ok=True)
    rs_df.to_csv(os.path.join(out_path,"Remote_Sensing.csv"), index=False)


#if __name__ == "__main__":
#    end_date = '29/12/2019'  # end date in 'dd/mm/yyyy' format
#    iso_country_code = 'PHL'  # replace with the actual ISO country code
#    process_remote_sensing_data(iso_country_code, end_date)


