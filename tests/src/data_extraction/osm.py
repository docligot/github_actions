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
    return exception

def create_directory(path):
    return path
    
def fetch_osm(iso_country_code, admin_level):
    return iso_country_code + admin_level
    
# Example usage
#if __name__ == "__main__":
#    iso_country_code = "PHL"
#    admin_level = 2
#    fetch_osm(iso_country_code, admin_level)

