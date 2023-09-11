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
    user = getpass.getpass(prompt='Enter NASA Earthdata Login Username: ')
    password = getpass.getpass(prompt='Enter NASA Earthdata Login Password: ')
    try:
        token_response = requests.post(f'{API_URL}login', auth=(user, password)).json()
        return {'Authorization': f'Bearer {token_response["token"]}'}
    except Exception as e:
        logging.error(f"Failed to get token: {e}")
        return None

def check_or_get_token(headers):
    if headers is None or 'Authorization' not in headers:
        return get_earthdata_token()
    return headers

def read_geojson(iso_country_code, adm_lvl, subregion):
    parent_folder = os.path.dirname(os.getcwd())
    geojson_file = os.path.join(parent_folder, 'data/GeoJSON', f"{iso_country_code}/ADM{adm_lvl}/{subregion}.geojson")
    with open(geojson_file, 'r') as f:
        return json.load(f)

def create_task_payload(iso_country_code, subregion, adm_lvl, start_date, end_date):
    geojson = read_geojson(iso_country_code, adm_lvl, subregion)
    return {
        'task_type': 'area',
        'task_name': f"{subregion}_HLS",
        'params': {
            'dates': [{'startDate': start_date, 'endDate': end_date}],
            'layers': [{'layer': f'B{str(i).zfill(2)}', 'product': 'HLSS30.020'} for i in range(1, 13)],
            'output': {'format': {'type': 'geotiff'}, 'projection': 'geographic'},
            'geo': geojson,
        }
    }

def create_task(headers, iso_country_code, subregion, adm_lvl, start_date, end_date):
    headers = check_or_get_token(headers)
    payload = create_task_payload(iso_country_code, subregion, adm_lvl, start_date, end_date)
    try:
        return requests.post(f'{API_URL}task', json=payload, headers=headers).json()
    except Exception as e:
        logging.error(f"Failed to create task: {e}")
        return None

def wait_for_task_completion(task_id, headers):
    headers = check_or_get_token(headers)
    starttime = time.time()
    while True:
        try:
            status = requests.get(f'{API_URL}task/{task_id}', headers=headers).json()['status']
            if status == 'done':
                return 'done'
            logging.info(f"Task status: {status}")
        except Exception as e:
            logging.error(f"Failed to get task status: {e}")
        time.sleep(SLEEP_INTERVAL - ((time.time() - starttime) % SLEEP_INTERVAL))

def download_files(task_id, headers, dest_dir):
    headers = check_or_get_token(headers)
    try:
        bundle = requests.get(f'{API_URL}bundle/{task_id}', headers=headers).json()

        # Create a list of bands you're interested in
        bands = [f'B{str(i).zfill(2)}' for i in range(1, 13)]
        
        # Filter only geotiff files of the spectral bands
        files = {f['file_id']: f['file_name'].split("/")[-1] for f in bundle['files'] if f['file_name'].endswith('.tif') and any(band in f['file_name'] for band in bands)}
        
        for file_id, file_name in files.items():
            dl = requests.get(f'{API_URL}bundle/{task_id}/{file_id}', headers=headers, stream=True)
            filepath = os.path.join(dest_dir, file_name)
            with open(filepath, 'wb') as f:
                for data in dl.iter_content(chunk_size=8192):
                    f.write(data)
        logging.info(f"Downloaded files can be found at: {dest_dir}")
    except Exception as e:
        logging.error(f"Failed to download files: {e}")

def fetch_nasa_appeears(headers, iso_country_code, subregion, adm_lvl, start_date, end_date):
    if not headers:
        logging.error("Headers are missing. Cannot proceed.")
        return

    task_response = create_task(headers, iso_country_code, subregion, adm_lvl, start_date, end_date)
    if task_response:
        task_id = task_response['task_id']
        if wait_for_task_completion(task_id, headers) == 'done':
            parent_folder = os.path.dirname(os.getcwd())
            dest_dir = os.path.join(parent_folder, 'data/NASA AppEEARS', f"{iso_country_code}/{subregion}/HLS")
            os.makedirs(dest_dir, exist_ok=True)
            download_files(task_id, headers, dest_dir)
    else:
        logging.error("Failed to create task. Cannot proceed.")

# Example usage        
#if __name__ == '__main__':
#    headers = get_earthdata_token()  # Initialize headers by fetching token once
#    fetch_nasa_appeears(headers, "PHL", "Zamboanga Sibugay", 2, '01-01-2021', '01-10-2021')

