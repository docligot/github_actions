#!/usr/bin/env python
# coding: utf-8

# In[20]:


from datetime import datetime
from pathlib import Path
import rasterio
import numpy as np
import os
import re
from skimage.filters import threshold_li
from collections import Counter
import geopandas as gpd
from shapely.geometry import Point
from collections import deque
from typing import List

def detect_hotspots(iso_country_code: str, subregion: str, start_date: str, end_date: str, duration_threshold: int) -> None:
    start_date_dt = datetime.strptime(start_date, '%d/%m/%Y')
    end_date_dt = datetime.strptime(end_date, '%d/%m/%Y')
    
    # Regular expression to find the Julian date in the filename
    date_pattern = re.compile(r'\d{7}')

    # Lists to store file paths
    ndvi_files = []
    ndwi_files = []

    # Iterate through prefixes and directories to find matching files
    for prefix in ["NDVI", "NDWI"]:
        path = os.path.join(os.path.dirname(os.getcwd()), 'data/NASA AppEEARS', f"{iso_country_code}/{subregion}/{prefix}")
        files = os.listdir(path)
        for file in files:
            match = date_pattern.search(file)
            if match:
                date_part = match.group()
                file_date = datetime.strptime(date_part, '%Y%j')
                if start_date_dt <= file_date <= end_date_dt:
                    if prefix == "NDVI":
                        ndvi_files.append(f"{path}/{file}")
                    else:
                        ndwi_files.append(f"{path}/{file}")
    
    # To store the shape of each file
    shapes = []

    # Iterate over time steps to get shapes
    for file in ndvi_files:
        with rasterio.open(file) as src:
            data = src.read(1)
        shapes.append(data.shape)

    # Calculate the modal (most common) shape
    modal_shape = Counter(shapes).most_common(1)[0][0]

    # Only keep files with the modal shape
    ndvi_files = [file for file in ndvi_files if rasterio.open(file).read(1).shape == modal_shape]
    ndwi_files = [file for file in ndwi_files if rasterio.open(file).read(1).shape == modal_shape]

    # Define your duration threshold
    duration_threshold = 4

    # Find matching filenames
    ndvi_filenames = {re.search(r'\d{7}', os.path.basename(f)).group(): f for f in ndvi_files}
    ndwi_filenames = {re.search(r'\d{7}', os.path.basename(f)).group(): f for f in ndwi_files}
    matching_files = set(ndvi_filenames.keys()) & set(ndwi_filenames.keys())
    matching_ndvi_files = [ndvi_filenames[file] for file in matching_files]
    matching_ndwi_files = [ndwi_filenames[file] for file in matching_files]

    # Initialize a 3D array to keep track of water detection
    # Use the shape of the first NDVI image for the array dimensions
    try:
        with rasterio.open(matching_ndvi_files[0]) as src:
            first_image_shape = src.read(1).shape
            transform = src.transform  # Get the transform here
    except FileNotFoundError:
        print("File not found.")
        exit()

    water_detected = np.zeros((len(matching_ndvi_files), *first_image_shape), dtype=np.bool_)

    # Initialize arrays to store NDWI variability
    ndwi_var_sparse = np.zeros((len(matching_ndvi_files), *first_image_shape))
    ndwi_var_moderate = np.zeros((len(matching_ndvi_files), *first_image_shape))
    ndwi_var_dense = np.zeros((len(matching_ndvi_files), *first_image_shape))

    # Iterate over time steps
    for t in range(len(matching_ndvi_files)):
        with rasterio.open(matching_ndvi_files[t]) as src:
            ndvi_data = src.read(1)
        with rasterio.open(matching_ndwi_files[t]) as src:
            ndwi_data = src.read(1)

        p2, p98 = np.percentile(ndvi_data, [2, 98])

        sparse_vegetation = ndvi_data < p2
        moderate_vegetation = (ndvi_data >= p2) & (ndvi_data <= p98)
        dense_vegetation = ndvi_data > p98

        threshold_sparse = np.percentile(ndwi_data[sparse_vegetation], 98)
        threshold_moderate = np.percentile(ndwi_data[moderate_vegetation], 98)
        threshold_dense = np.percentile(ndwi_data[dense_vegetation], 98)

        water_detected[t, sparse_vegetation] = ndwi_data[sparse_vegetation] > threshold_sparse
        water_detected[t, moderate_vegetation] = ndwi_data[moderate_vegetation] > threshold_moderate
        water_detected[t, dense_vegetation] = ndwi_data[dense_vegetation] > threshold_dense

        # Store NDWI values for variability calculation
        ndwi_var_sparse[t, sparse_vegetation] = ndwi_data[sparse_vegetation]
        ndwi_var_moderate[t, moderate_vegetation] = ndwi_data[moderate_vegetation]
        ndwi_var_dense[t, dense_vegetation] = ndwi_data[dense_vegetation]

    # Calculate the standard deviation of NDWI across time
    std_dev_sparse = np.std(ndwi_var_sparse, axis=0)
    std_dev_moderate = np.std(ndwi_var_moderate, axis=0)
    std_dev_dense = np.std(ndwi_var_dense, axis=0)

    # Use Li's method to find the threshold for NDWI variability
    std_dev_threshold_sparse = threshold_li(std_dev_sparse)
    std_dev_threshold_moderate = threshold_li(std_dev_moderate)
    std_dev_threshold_dense = threshold_li(std_dev_dense)

    # Create masks for each vegetation type
    mask_sparse = np.zeros(first_image_shape, dtype=np.bool_)
    mask_moderate = np.zeros(first_image_shape, dtype=np.bool_)
    mask_dense = np.zeros(first_image_shape, dtype=np.bool_)

    mask_sparse[sparse_vegetation] = True
    mask_moderate[moderate_vegetation] = True
    mask_dense[dense_vegetation] = True

    # Update water detection based on NDWI variability
    for t in range(len(matching_ndvi_files)):
        water_detected[t, mask_sparse] &= std_dev_sparse[mask_sparse] < std_dev_threshold_sparse
        water_detected[t, mask_moderate] &= std_dev_moderate[mask_moderate] < std_dev_threshold_moderate
        water_detected[t, mask_dense] &= std_dev_dense[mask_dense] < std_dev_threshold_dense

    # Initialize a 2D array to keep track of stagnant water
    stagnant_water = np.zeros(first_image_shape, dtype=np.bool_)

    # Iterate over each pixel
    for i in range(water_detected.shape[1]):
        for j in range(water_detected.shape[2]):

            # Use a deque to efficiently keep track of the last few time steps
            recent_detections = deque(maxlen=duration_threshold)

            # Iterate over time steps for this pixel
            for t in range(water_detected.shape[0]):
                recent_detections.append(water_detected[t, i, j])

                # If water was detected in all of the last few time steps, mark as stagnant
                if np.all(recent_detections): 
                    stagnant_water[i, j] = True

    # Export stagnant water data
    geometries = []
    for i in range(stagnant_water.shape[0]):
        for j in range(stagnant_water.shape[1]):
            if stagnant_water[i, j]:
                lon, lat = transform * (j, i)
                 # Truncate lat lon up to 5 decimal places
                lon = round(lon, 3)
                lat = round(lat, 3)
                geometries.append(Point(lon, lat))

    gdf = gpd.GeoDataFrame(geometry=geometries)
    gdf = gdf.drop_duplicates(subset='geometry')
    
    # Export stagnant water data
    # Create the directory if it doesn't exist
    gdf_dir = Path(os.path.dirname(os.getcwd())) / 'processed/Dashboard/Hotspot Detection' / iso_country_code
    os.makedirs(gdf_dir, exist_ok=True)

    # Define the full path for the GeoJSON file
    gdf_path = gdf_dir / f"{subregion}.geojson"
    gdf.to_file(gdf_path, driver='GeoJSON')

# Example usage
#detect_hotspots("PHL", "Zamboanga Sibugay", '18/03/2023', '18/06/2023', 8)

