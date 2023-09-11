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
    print(f"Query sent: {query}")  # Debugging line
    url = f"https://geocode.maps.co/search?q={query}"
    retries = 3  # Number of retries
    delay = 5  # Delay in seconds
    
    while retries > 0:
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                data = json.loads(response.text)
                
                # Debug: Print the data
                #print("Received data:", data)
                
                if len(data) > 0:
                    # Initialize default lat and lon with the first entry
                    default_lat = data[0].get('lat', None)
                    default_lon = data[0].get('lon', None)
                    
                    for entry in data:
                        if entry.get('class') == 'boundary' and entry.get('type') == 'administrative':
                            lat = entry['lat']
                            lon = entry['lon']
                            return f"{lat},{lon}"
                    
                    # If no entry with class: boundary and type: administrative is found
                    if default_lat is not None and default_lon is not None:
                        return f"{default_lat},{default_lon}"
                else:
                    print("Data list is empty.")
                    return None
            elif response.status_code == 429:
                print("Rate limit exceeded. Cooling down for 1 second.")
                time.sleep(1)
            elif response.status_code == 503:
                retry_after = int(response.headers.get("Retry-After", 1))
                print(f"Server busy. Cooling down for {retry_after} seconds.")
                time.sleep(retry_after)
            elif response.status_code == 403:
                print("Access forbidden. Please contact the service provider.")
                return None
            else:
                print("An unexpected error occurred.")
                return None
            
            time.sleep(0.5)  # To ensure we don't exceed the rate limit
        except (requests.ConnectionError, requests.Timeout) as e:
            print(f"Error occurred: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
            retries -= 1
    
    print("Max retries reached. Could not complete the request.")
    return None

# Function to generate bounding box
def latlong_gen(latlong, diff):
    lat_v, long_v = latlong.split(",")
    LL_lat = round(float(lat_v) - diff, 4)
    LL_long = round(float(long_v) - diff, 4)
    UR_lat = round(float(lat_v) + diff, 4)
    UR_long = round(float(long_v) + diff, 4)
    bbox = f"{LL_lat},{LL_long},{UR_lat},{UR_long}"
    return bbox

# Function to generate URL
def url_gen(full_date_txt, bbox, layers, width, height, wrap):
    epoch_time = int(datetime.datetime.now().timestamp())
    url = f"https://wvs.earthdata.nasa.gov/api/v1/snapshot?REQUEST=GetSnapshot&TIME={full_date_txt}T00:00:00Z&BBOX={bbox}&CRS=EPSG:4326&LAYERS={layers}&WRAP={wrap}&FORMAT=image/jpeg&WIDTH={width}&HEIGHT={height}&ts={epoch_time}"
    return url

def next_date(current_date, interval):
    if interval == 'D':
        return current_date + timedelta(days=1)
    elif interval == 'W':
        return current_date + timedelta(weeks=1)
    elif interval == 'M':
        return current_date + timedelta(days=30)  # Approximation
    elif interval == 'Q':
        return current_date + timedelta(days=90)  # Approximation
    else:
        return None

# ... (rest of the code remains the same)

def fetch_nasa_worldview(iso_country_code, subregion, region, start_date, end_date, interval):
    layers = {
                    'fapar': ['Coastlines_15m,MODIS_Combined_L4_FPAR_4Day,MODIS_Terra_L4_FPAR_8Day,MODIS_Aqua_L4_FPAR_8Day,MODIS_Combined_L4_FPAR_8Day', 'x,none,none,none,none'],
                    'evi' : ['Coastlines_15m,MODIS_Aqua_L3_EVI_16Day','x,none'],
                    'ndvi' : ['Coastlines_15m,MODIS_Terra_L3_NDVI_16Day','x,none'],
                    'lst_m_day' : ['Coastlines_15m,MODIS_Terra_L3_Land_Surface_Temp_Monthly_Day', 'x,none'],
                    'lst_m_nt' : ['Coastlines_15m,MODIS_Terra_L3_Land_Surface_Temp_Monthly_Night', 'x,none'],
                    'precip' : ['Coastlines_15m,IMERG_Precipitation_Rate', 'x,none'],
                    'elev' : ['Coastlines_15m,ASTER_GDEM_Color_Index', 'x,none'],

            # Add more layers here
        }
    # Fixed width, height, and diff
    width = 800
    height = 800
    diff = 0.1103
    
    start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    
    query = f"{subregion}+{region}"
    query = query.strip().replace(" ", "+")
    latlong = get_lat_lon(query)
    print(latlong)
    
    if latlong:
        bbox = latlong_gen(latlong, diff)
        for layer_name, layer_wrap in layers.items():
            layer = layer_wrap[0]
            wrap = layer_wrap[1]  # Extract the wrap value
            
            # Modified directory path as you specified
            parent_dir = os.path.dirname(os.getcwd())
            directory = os.path.join(parent_dir, 'data/Worldview', f"{iso_country_code}/{subregion}")
            path = os.path.join(directory, layer_name)
            
            os.makedirs(path, exist_ok=True)
            
            current_date = start_date
            
            while current_date <= end_date:
                try:
                    full_date_txt = current_date.strftime("%Y-%m-%d")
                    url = url_gen(full_date_txt, bbox, layer, width, height, wrap)  # Include wrap here
                    file_name = wget.download(url, out=path)
                    print(f'Image Successfully Downloaded: {file_name}')
                except Exception as e:
                    print(f"Error for date: {full_date_txt}. Exception: {e}")
                
                current_date = next_date(current_date, interval)

# Example usage
#if __name__ == "__main__":
#    fetch_nasa_worldview("PHL", "Zamboanga Sibugay", "Zamboanga Peninsula", "2016-01-10", "2021-01-10", "W")

