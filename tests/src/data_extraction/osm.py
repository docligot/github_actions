#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import osmnx as ox
import geopandas as gpd
import pandas as pd
from datetime import datetime
from retrying import retry

# Update this line to use the new settings module
ox.settings.timeout = 60

def retry_if_file_not_found_error(exception):
    return isinstance(exception, FileNotFoundError)

def create_directory(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)

@retry(retry_on_exception=retry_if_file_not_found_error, stop_max_attempt_number=3, wait_fixed=2000)
def fetch_osm(iso_country_code: str, admin_level: int) -> None:
    src_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'GeoJSON', f"{iso_country_code}/ADM{admin_level}")
    output_dir = os.path.join(os.path.dirname(os.getcwd()), 'data', 'OpenStreetMap', f"{iso_country_code}/ADM{admin_level}")

    create_directory(output_dir)

    admin_list = [f.split('.geojson')[0] for f in os.listdir(src_path) if f.endswith('.geojson')]

    # Initialize data frames
    tag_dfs = {"building": pd.DataFrame(), "amenity": pd.DataFrame(), "shop": pd.DataFrame(), "water": pd.DataFrame()}

    # Load any existing temporary data
    for tag in tag_dfs.keys():
        temp_output_path = os.path.join(output_dir, f"{tag}_temp.csv")
        if os.path.exists(temp_output_path):
            tag_dfs[tag] = pd.read_csv(temp_output_path)

    # Read the last processed location from a file
    last_processed_file = os.path.join(output_dir, "last_processed.txt")
    if os.path.exists(last_processed_file):
        with open(last_processed_file, 'r') as f:
            last_processed = f.read().strip()
        start_index = admin_list.index(last_processed) + 1 if last_processed in admin_list else 0
    else:
        start_index = 0

    for loc in admin_list[start_index:]:
        flnm = os.path.join(src_path, f"{loc}.geojson")
        
        try:
            gdf = gpd.read_file(flnm)
        except FileNotFoundError:
            print(f"File {flnm} not found. Retrying...")
            continue

        geometry = gdf['geometry'][0]
        
        for tag in tag_dfs.keys():
            try:
                df = ox.features_from_polygon(geometry, tags={tag: True})
                cnt_df = pd.DataFrame(df[tag].value_counts()).reset_index().rename(columns={"index": tag, tag: "count"}).T
                cnt_df2 = cnt_df.drop(index=[tag])
                cnt_df2.columns = cnt_df.iloc[0].values.tolist()
                cnt_df2["Location"] = str(loc)
                cnt_df2["Date_Extracted"] = datetime.now().strftime('%Y-%m-%d')
                
                tag_dfs[tag] = tag_dfs[tag].append(cnt_df2, ignore_index=True)
                
            except Exception as e:
                print(f"No {tag} found for {loc}. Error: {e}")
                continue

        print(f"Processing for {loc} is complete.")

        # Save intermediate data
        for tag, df in tag_dfs.items():
            temp_output_path = os.path.join(output_dir, f"{tag}_temp.csv")
            df.to_csv(temp_output_path, index=False)

        # Update the last processed location
        with open(last_processed_file, 'w') as f:
            f.write(loc)

    # Save final data and remove temporary files
    for tag, df in tag_dfs.items():
        output_path = os.path.join(output_dir, f"{tag}.csv")
        df.to_csv(output_path, index=False)
        os.remove(os.path.join(output_dir, f"{tag}_temp.csv"))
        print(f"{tag.capitalize()} data has been saved to {output_path}")

    # Remove the last processed file as all locations have been processed
    if os.path.exists(last_processed_file):
        os.remove(last_processed_file)

# Example usage
#if __name__ == "__main__":
#    iso_country_code = "PHL"
#    admin_level = 2
#    fetch_osm(iso_country_code, admin_level)

