#!/usr/bin/env python
# coding: utf-8

# In[12]:


import requests
import wget
import datetime
import os
import json
import time
from datetime import timedelta

# Function to get latitude and longitude
def get_lat_lon(query):
    return query

# Function to generate bounding box
def latlong_gen(latlong, diff):
    return latlong + diff
    
# Function to generate URL
def url_gen(full_date_txt, bbox, layers, width, height, wrap):
    return full_date_txt + bbox + layers + width + height + wrap
    
    
def next_date(current_date, interval):
    return current_date + interval
# ... (rest of the code remains the same)

def fetch_nasa_worldview(iso_country_code, subregion, region, start_date, end_date, interval):
    return iso_country_code + subregion + region + start_date + end_date + interval

# Example usage
#if __name__ == "__main__":
#    fetch_nasa_worldview("PHL", "Zamboanga Sibugay", "Zamboanga Peninsula", "2016-01-10", "2021-01-10", "W")

