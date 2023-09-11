#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
from datetime import datetime
import rasterio
import numpy as np
import re

def hls_to_indices(
    iso_country_code: str, 
    subregion: str, 
    index_func: callable,
    start_date_str: str,
    end_date_str: str    
) -> None:
    
    # Convert date strings to datetime objects
    START_DATE = datetime.strptime(start_date_str, '%d/%m/%Y')
    END_DATE = datetime.strptime(end_date_str, '%d/%m/%Y')

    # Create a regular expression to find the Julian date in the filename
    DATE_PATTERN = re.compile(r'\d{7}')

    BASE_PATH = os.path.join(os.path.dirname(os.getcwd()), 'data/NASA AppEEARS', f"{iso_country_code}/{subregion}/HLS")
    output_path = index_func.__name__.split('_')[1].upper()  # Extracting the index name from the function name and converting to uppercase
    out_dir = os.path.join(os.path.dirname(os.getcwd()), 'data/NASA AppEEARS', f"{iso_country_code}/{subregion}", output_path)
    os.makedirs(out_dir, exist_ok=True)
    
    try:
        files = os.listdir(BASE_PATH)
    except FileNotFoundError:
        print(f"No files found in directory: {BASE_PATH}")
        return

    # Filter files based on prefix and date
    prefixes = ['HLSS30.020_B03', 'HLSS30.020_B04', 'HLSS30.020_B08', 'HLSS30.020_B11']
    band_files = {prefix: {} for prefix in prefixes}

    for file in files:
        for prefix in prefixes:
            if file.startswith(prefix):
                match = DATE_PATTERN.search(file)
                if match:
                    date_part = match.group()
                    file_date = datetime.strptime(date_part, '%Y%j')
                    if START_DATE <= file_date <= END_DATE:
                        band_files[prefix][date_part] = file

    for date in band_files[prefixes[0]]:
        if date in band_files[prefixes[1]]:
            file1 = band_files[prefixes[0]][date]
            file2 = band_files[prefixes[1]][date]

            try:
                with rasterio.open(os.path.join(BASE_PATH, file1)) as src1:
                    band1 = src1.read(1)
                    crs1 = src1.crs
                    transform1 = src1.transform

                with rasterio.open(os.path.join(BASE_PATH, file2)) as src2:
                    band2 = src2.read(1)
                    crs2 = src2.crs
                    transform2 = src2.transform

                if crs1 != crs2 or transform1 != transform2:
                    print(f"Files for date {date} have different CRS or transform.")
                    continue

                if np.all(band1 == -9999) and np.all(band2 == -9999):
                    continue

                index = index_func(band1, band2)
                if np.any(index == np.inf) or np.any(index == -np.inf):
                    index[band1 + band2 == 0] = -9999

                filename = f"HLSS30.020_{output_path}_doy{date}_aid0001_51N.tif"
                with rasterio.open(
                    os.path.join(out_dir, filename),
                    'w',
                    driver='GTiff',
                    height=index.shape[0],
                    width=index.shape[1],
                    count=1,
                    dtype=index.dtype,
                    crs=src1.crs,
                    transform=src1.transform,
                    nodata=-9999
                ) as dst:
                    dst.write(index, 1)

            except Exception as e:
                print(f"Error processing files for date {date}: {e}")

def calculate_ndvi(band1: np.ndarray, band2: np.ndarray) -> np.ndarray:
    return (band2.astype(float) - band1.astype(float)) / (band2.astype(float) + band1.astype(float)) * 10000

def calculate_ndwi(band1: np.ndarray, band2: np.ndarray) -> np.ndarray:
    return (band1.astype(float) - band2.astype(float)) / (band1.astype(float) + band2.astype(float)) * 10000

def calculate_ndbi(band1: np.ndarray, band2: np.ndarray) -> np.ndarray:
    return (band2.astype(float) - band1.astype(float)) / (band2.astype(float) + band1.astype(float)) * 10000

def calculate_fpar(band1: np.ndarray, band2: np.ndarray) -> np.ndarray:
    ndvi = (band2.astype(float) - band1.astype(float)) / (band2.astype(float) + band1.astype(float))
    ndvi[band1 + band2 == 0] = -9999
    fpar = np.where(ndvi < 0, 0, np.where(ndvi > 1, 1, 0.57 * np.power(ndvi, 2.63))) * 10000
    fpar[ndvi == -9999] = -9999
    return fpar

# Example usage
#iso_code = "PHL"  # Example ISO country code for Philippines
#subregion_name = "Zamboanga Sibugay"  # Example subregion
#hls_to_indices(iso_code, subregion_name, calculate_ndvi, '01/12/2019', '18/06/2023')
#hls_to_indices(iso_code, subregion_name, calculate_ndwi, '01/12/2019', '18/06/2023')
#hls_to_indices(iso_code, subregion_name, calculate_ndbi, '01/12/2019', '18/06/2023')
#hls_to_indices(iso_code, subregion_name, calculate_fpar, '01/12/2019', '18/06/2023')

