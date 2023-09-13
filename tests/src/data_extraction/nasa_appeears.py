#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import getpass
import json
import os
import time
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)

API_URL = 'https://appeears.earthdatacloud.nasa.gov/api/'
SLEEP_INTERVAL = 20

def get_earthdata_token():
    return None

def check_or_get_token(headers):
    return headers

def read_geojson(iso_country_code, adm_lvl, subregion):
    return iso_country_code + adm_lvl + subregion
    
def create_task_payload(iso_country_code, subregion, adm_lvl, start_date, end_date):
    return iso_country_code + subregion + adm_lvl + start_date + end_date

def create_task(headers, iso_country_code, subregion, adm_lvl, start_date, end_date):
    return headers + iso_country_code + subregion + adm_lvl + start_date + end_date
    
def wait_for_task_completion(task_id, headers):
    return task_id + headers
    
def download_files(task_id, headers, dest_dir):
    return task_id + headers + dest_dir

def fetch_nasa_appeears(headers, iso_country_code, subregion, adm_lvl, start_date, end_date):
    return headers + iso_country_code + subregion + adm_lvl + start_date + end_date
    
# Example usage        
#if __name__ == '__main__':
#    headers = get_earthdata_token()  # Initialize headers by fetching token once
#    fetch_nasa_appeears(headers, "PHL", "Zamboanga Sibugay", 2, '01-01-2021', '01-10-2021')

