#!/usr/bin/env python
# coding: utf-8

# In[4]:


import requests
import re
import os

def fetch_relative_wealth_index(country, iso_country_code):
    url = "https://data.humdata.org/dataset/relative-wealth-index"
    response = requests.get(url)
    
    if response.status_code != 200:
        print("Failed to access the website.")
        return
    
    # Use regular expression to find the URL-like text
    pattern = re.compile(r'https://data\.humdata\.org/dataset/[\w\d-]+/resource/[\w\d-]+/download/{}_relative_wealth_index\.csv'.format(iso_country_code.lower()))
    match = pattern.search(response.text)
    
    if match:
        download_url = match.group(0)
        download_rwi(download_url, country, iso_country_code)
    else:
        print(f"No data found for {country}")

def download_rwi(url, country, iso_country_code):
    response = requests.get(url)
    if response.status_code == 200:
        file_dir = os.path.join(os.path.dirname(os.getcwd()), f'data/Demographic/{iso_country_code}')
        os.makedirs(file_dir, exist_ok=True)
        file_path = os.path.join(file_dir, f"{iso_country_code}_relative_wealth_index.csv")
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f"File downloaded: {file_path}")
    else:
        print("Failed to download the file.")


# In[6]:


def fetch_population_density(country, iso_country_code, segment):
    url = "https://data.humdata.org/dataset/philippines-high-resolution-population-density-maps-demographic-estimates"
    response = requests.get(url)
    
    if response.status_code != 200:
        print("Failed to access the website.")
        return
    
    # Use a simplified regular expression to find the URL-like text
    pattern = re.compile(r'https://.*?download/{}_{}_.*_geotiff\.zip'.format(iso_country_code.lower(), segment))
    match = pattern.search(response.text)
    
    if match:
        download_url = match.group(0)
        print(f"Beginning download for {country} and segment {segment}. This may take a while due to file size.")
        download_popmap(download_url, country, iso_country_code, segment)
    else:
        print(f"No data found for {country} and segment {segment}")

def download_popmap(url, country, iso_country_code, segment):
    response = requests.get(url)
    if response.status_code == 200:
        file_dir = os.path.join(os.path.dirname(os.getcwd()), f'data/Demographic/{iso_country_code}')
        os.makedirs(file_dir, exist_ok=True)
        file_path = os.path.join(file_dir, f"{iso_country_code}_{segment}_population_density.zip")
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f"File downloaded: {file_path}")
    else:
        print("Failed to download the file.")

# Example usage
#fetch_population_density("Philippines", "PHL", "general")
# Example usage
#fetch_relative_wealth_index("Philippines", "PHL")

