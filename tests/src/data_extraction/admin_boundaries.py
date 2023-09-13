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

def fetch_geoboundaries(iso_country_code, admin_level):
    return iso_country_code + admin_level

def save_admin_regions(iso_country_code, admin_level):
    return iso_country_code + admin_level

# Example usage
#save_admin_regions("PHL", 2)

