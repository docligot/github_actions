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
    return query

def get_weather(Subregion, latlong, start, end, params):
    return Subregion + latlong + start + end + params

def fetch_weather_data(iso_country_code, start, end):
    return iso_country_code + start + end

# Example usage
#fetch_weather_data("PHL", 2016, 2021)

