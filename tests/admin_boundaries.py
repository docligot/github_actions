#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import json
import requests
import geopandas as gpd
from fiona.crs import from_epsg
from time import sleep
import warnings

warnings.filterwarnings("ignore")

class Admin_boundaries: 

    @staticmethod
    def fetch_geoboundaries(iso_country_code, admin_level):
        api_url = f"https://www.geoboundaries.org/gbRequest.html?ISO={iso_country_code}&ADM=ADM{admin_level}"
        print(api_url)
        
        retries = 3
        for i in range(retries):
            try:
                response = requests.get(api_url, timeout=20)
                
                if response.status_code != 200:
                    print(f"Failed to fetch data from GeoBoundaries API. Status code: {response.status_code}")
                    return None
                
                dlPath = response.json()[0]['gjDownloadURL']
                print(dlPath)
                geoBoundary = requests.get(dlPath, timeout=20).json()
                
                return gpd.GeoDataFrame.from_features(geoBoundary["features"])
            
            except requests.exceptions.RequestException as e:
                print(f"An error occurred: {e}")
                print("Retrying...")
                sleep(5)  # wait for 5 seconds before retrying
        
        print("Max retries reached. Exiting.")
        return None

    @staticmethod
    def save_admin_regions(iso_country_code, admin_level):
        geoBoundary = fetch_geoboundaries(iso_country_code, admin_level)
        if geoBoundary is None:
            print("Received empty data from API.")
            return

        # Create 'data/GeoJSON' folder if it doesn't exist
        data_folder = os.path.join(os.path.dirname(os.getcwd()),'data', 'GeoJSON', f"{iso_country_code}/ADM{admin_level}")
        os.makedirs(data_folder, exist_ok=True)

        for index, row in geoBoundary.iterrows():
            shape_name = row['shapeName']
            gdf = geoBoundary.loc[[index]]
            
            # Save to GeoJSON file
            flnm = os.path.join(data_folder, f"{shape_name}.geojson")
            gdf.to_file(flnm, driver='GeoJSON')

# Example usage
#save_admin_regions("PHL", 2)

