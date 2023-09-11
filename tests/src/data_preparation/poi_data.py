#!/usr/bin/env python
# coding: utf-8

# In[18]:


import os
import pandas as pd
import numpy as np
from functools import reduce
from typing import List

def process_osm_data(iso_country_code: str, adm_level: int) -> None:
    """
    Process OpenStreetMap data and save it to a CSV file.

    Parameters:
    - iso_country_code (str): ISO country code.
    - adm_level (int): Administrative level.

    Returns:
    None
    """
    # Initialize source directory and file paths
    src_dir = os.path.join(os.path.dirname(os.getcwd()), 'data/OpenStreetMap', f"{iso_country_code}/ADM{adm_level}")

    # Read and preprocess the data
    combined_df = pd.DataFrame()
    for file_name, columns in [("building.csv", ['Location','house', 'apartments', 'residential']),
                               ("amenity.csv", ["Location","toilets","waste_disposal","waste_transfer_station",'school', \
                                      'university', 'college', 'kindergarten', 'childcare', 'prep_school', \
                                      'dormitory', 'nursing_home', 'quarantine_facility', 'birthing_centre', \
                                      'hospital', 'clinic', 'health_post', 'healthcare', 'pharmacy', 'doctors', \
                                      'rescue-station', 'rescue', 'community_centre', 'social_centre', 'social_facility', \
                                      'emergency_service']),
                               ("water.csv", ["Location",'basin', 'canal', 'ditch', 'drain', 'fish_pass', 'harbour', 'lock', 'moat', 'pond', \
                                'reflecting_pool', 'reservoir', 'wastewater', 'lagoon', 'lake', 'oxbow', 'rapids', \
                                'river', 'stream', 'stream_pool'])]:
        df = pd.read_csv(os.path.join(src_dir, file_name))
        df["Location"] = df["Location"].apply(lambda l: l.upper().strip())
        df = df[[col for col in columns if col in df.columns]]
        if len(combined_df) == 0:
            combined_df = combined_df.append(df)
        else:
            combined_df = combined_df.merge(df, on="Location", how="left")
    

    # Save the processed data
    out_path = os.path.join(os.path.dirname(os.getcwd()), 'processed/INFORM', iso_country_code)
    os.makedirs(out_path, exist_ok=True)
    combined_df.to_csv(os.path.join(out_path, "POI.csv"), index=False)

# Example usage    
#if __name__ == "__main__":
#    process_osm_data("PHL", 2)

