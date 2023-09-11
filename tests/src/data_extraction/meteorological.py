#!/usr/bin/env python
# coding: utf-8

# In[2]:


import requests
import pandas as pd
import io
import time
import random                                     
import os
import json

def get_lat_lon(query):
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

def get_weather(Subregion, latlong, start, end, params):
    city = Subregion.upper()
    latitude, longitude = latlong.split(",")
    latitude = float(latitude.replace("'", ""))
    longitude = float(longitude.replace("'", ""))
    
    req_url = 'https://power.larc.nasa.gov/api/temporal/daily/point?' 
    req_query = {
        'parameters': params,
        'community': 'ag',
        'latitude': latitude,
        'longitude': longitude,
        'start': start,
        'end': end,
        'format': 'csv',
        'header': 'false'
    }    

    data = requests.get(req_url, params=req_query).content.decode('utf8')
    df = pd.read_csv(io.StringIO(data))
    df['loc'] = city
    print('Scraping done for:', city)
    return df

def fetch_weather_data(iso_country_code, start, end):
    csv_path = os.path.join(os.path.dirname(os.getcwd()), f"data\Admin_Boundaries_{iso_country_code}.csv")
    df = pd.read_csv(csv_path)
    df['Subregion+Region'] = df['Subregion'] + "+" + df['Region']
    df['Subregion+Region'] = df['Subregion+Region'].str.replace(" ", "+")
    df['latlong'] = df['Subregion+Region'].apply(get_lat_lon)

    params = 'TS,T2M,QV2M,RH2M,T2MDEW,T2MWET,GWETTOP,T2M_MAX,T2M_MIN,GWETPROF,GWETROOT,CLOUD_AMT,T2M_RANGE,PRECTOTCORR,ALLSKY_SFC_LW_DWN'
    new_df = pd.DataFrame()

    for Subregion, latlong in zip(df['Subregion'].tolist(), df['latlong'].tolist()):
        if latlong is not None:
            if len(new_df) == 0:
                new_df = get_weather(Subregion, latlong, start, end, params)
            else:
                new_df = new_df.append(get_weather(Subregion, latlong, start, end, params))
        
            time.sleep(random.uniform(1, 5))
            
    output_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'NASA POWER', f"{iso_country_code}.csv")
    new_df.to_csv(output_path, index=False)

# Example usage
#fetch_weather_data("PHL", 2016, 2021)

