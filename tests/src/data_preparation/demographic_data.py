#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.windows import from_bounds
from rasterio.warp import transform as rasterio_transform
from shapely.geometry import Point
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

def load_geotiff_paths(directory: str) -> Dict[str, str]:
    """
    Load GeoTIFF paths from a directory into a dictionary.

    Parameters:
    - directory (str): The directory containing the GeoTIFF files.

    Returns:
    Dict[str, str]: A dictionary mapping labels to GeoTIFF file paths.
    """
    geotiff_paths = {}
    for filename in os.listdir(directory):
        if filename.endswith('.tif'):
            label = filename.split('_')[1]  # Extract the label from the filename
            if filename == "phl_general_2020.tif":  # Special case for 'phl_general_2020.tif'
                label = "total"
            full_path = os.path.join(directory, filename)
            geotiff_paths[label] = full_path
    return geotiff_paths

def compute_population_metrics(geojson_path: str, geotiff_path: str) -> Tuple[float, float, float]:
    """
    Compute population metrics based on GeoJSON and GeoTIFF files.
    
    Parameters:
    - geojson_path (str): Path to the GeoJSON file.
    - geotiff_path (str): Path to the GeoTIFF file.
    
    Returns:
    Tuple[float, float, float]: Total population, average density, and region area in km^2.
    """
    # Read the GeoJSON file into a GeoDataFrame
    geo_df = gpd.read_file(geojson_path)
    
    # Check if CRS is set; if not, assign it
    if geo_df.crs is None:
        geo_df.set_crs(epsg=4326, inplace=True)
    else:
        geo_df = geo_df.to_crs(epsg=4326)

    # Convert to a projected CRS for area calculation
    geo_df_projected = geo_df.to_crs(epsg=32651)
    region_area_m2 = abs(geo_df_projected.geometry.area.sum())  # Area in square meters
    region_area_km2 = region_area_m2 / 1_000_000  # Convert to square kilometers

    with rasterio.open(geotiff_path) as src:
        # Convert the bounding box of the GeoJSON to the coordinate system of the GeoTIFF
        bounds = geo_df.geometry.bounds.iloc[0]
        
        # Transform the bounding box
        src_crs = src.crs.to_string()
        transformed_x, transformed_y = rasterio_transform(
            geo_df.crs, 
            src_crs, 
            [bounds.minx, bounds.maxx], 
            [bounds.miny, bounds.maxy]
        )
        minx, maxx = transformed_x
        miny, maxy = transformed_y
        
        # Define the window to read based on the transformed bounding box
        window = from_bounds(minx, miny, maxx, maxy, src.transform)
        out_image = src.read(window=window)
        out_transform = src.window_transform(window)
    
        x_res = out_transform[0]
        y_res = out_transform[4]
        pixel_area = abs(x_res * y_res)
        
        total_population = np.nansum(out_image)
        average_density = total_population / region_area_km2
    
    return total_population, average_density, region_area_km2

from typing import Dict
import geopandas as gpd

def compute_rwi_metrics(geojson_path: str, wealth_gdf: gpd.GeoDataFrame) -> Dict[str, float]:
    """
    Compute Relative Wealth Index (RWI) metrics based on a GeoJSON file and wealth data.
    
    Parameters:
    - geojson_path (str): Path to the GeoJSON file.
    - wealth_gdf (gpd.GeoDataFrame): GeoDataFrame containing wealth data.
    
    Returns:
    Dict[str, float]: A dictionary containing the average, minimum, and maximum RWI.
    """
    # Read the GeoJSON file into a GeoDataFrame
    geom = gpd.read_file(geojson_path)
    
    # Determine which points lie within the geometry
    points_within_geom = wealth_gdf[wealth_gdf.geometry.within(geom.geometry[0])]
    
    # Calculate the average, minimum, and maximum RWI values for those points
    average_rwi = points_within_geom['rwi'].mean()
    min_rwi = points_within_geom['rwi'].min()
    max_rwi = points_within_geom['rwi'].max()
    
    return {
        'Average RWI': average_rwi,
        'Min RWI': min_rwi,
        'Max RWI': max_rwi
    }

def process_demogs(iso_country_code: str, adm_level: int) -> None:
    """Main function to compute and save demographic data."""
    # Initialize paths and lists
    geojson_path = os.path.join(os.path.dirname(os.getcwd()), 'data/GeoJSON', f"{iso_country_code}/ADM{adm_level}")
    loc_list = [f.split(".")[0] for f in os.listdir(geojson_path)]
    
    # Load GeoTIFF paths
    directory = os.path.join(os.path.dirname(os.getcwd()), 'data/Demographic', f"{iso_country_code}")
    #directory = r"D:\aedes\02_Data"
    geotiff_paths = load_geotiff_paths(directory)
    
    # Initialize results
    popn_results = []
    rwi_results = []
    
    # Load wealth data
    wealth_path = os.path.join(os.path.dirname(os.getcwd()), 'data/Demographic', f"{iso_country_code}")
    wealth_data = pd.read_csv(os.path.join(wealth_path, f"{iso_country_code}_relative_wealth_index.csv"))
    geometry = [Point(xy) for xy in zip(wealth_data['longitude'], wealth_data['latitude'])]
    wealth_gdf = gpd.GeoDataFrame(wealth_data, geometry=geometry)
    
    for loc in loc_list:
        flnm = os.path.join(geojson_path, f"{loc}.geojson")
        
        # Compute population metrics
        popn_row = {'Location': loc}
        _, _, area_km2 = compute_population_metrics(flnm, list(geotiff_paths.values())[0])
        popn_row['Area (km^2)'] = area_km2
        for key, geotiff_path in geotiff_paths.items():
            total_population, average_density, _ = compute_population_metrics(flnm, geotiff_path)
            popn_row[f"{key} population"] = total_population
            popn_row[f"{key} density"] = average_density
        popn_results.append(popn_row)
        
        # Compute RWI metrics
        rwi_row = {'Location': loc}
        rwi_metrics = compute_rwi_metrics(flnm, wealth_gdf)
        rwi_row.update(rwi_metrics)
        rwi_results.append(rwi_row)
    
    # Create and save DataFrames
    popn_df = pd.DataFrame(popn_results)
    rwi_df = pd.DataFrame(rwi_results)
    final_df = popn_df.merge(rwi_df, on="Location", how="left")
    
    out_path = os.path.join(os.path.dirname(os.getcwd()), 'processed/INFORM', f"{iso_country_code}")
    os.makedirs(out_path, exist_ok=True)
    final_df.to_csv(os.path.join(out_path, "Demographics.csv"), index=False)

# Example usage    
#if __name__ == "__main__":
#    process_demogs("PHL", 2)

