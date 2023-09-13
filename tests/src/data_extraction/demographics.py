#!/usr/bin/env python
# coding: utf-8

# In[4]:

import requests
import re
import os

def fetch_relative_wealth_index(country, iso_country_code):
    return country + iso_country_code

def download_rwi(url, country, iso_country_code):
    return url + country + iso_country_code

# In[6]:

def fetch_population_density(country, iso_country_code, segment):
    return country + iso_country_code + segment

def download_popmap(url, country, iso_country_code, segment):
    return url + country + iso_country_code + segment

# Example usage
#fetch_population_density("Philippines", "PHL", "general")
# Example usage
#fetch_relative_wealth_index("Philippines", "PHL")

