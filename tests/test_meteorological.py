import requests
import pandas as pd
import io
import time
import random                                     
import os
import json

from .src.data_extraction.meteorological import get_lat_lon, get_weather, fetch_weather_data

def test_get_lan_lon():
    assert get_lat_lon(1) == 1

def test_get_weather():
    assert get_weather(1, 2, 3, 4, 5) == 15
    
def test_fetch_weather_data(): 
    assert fetch_weather_data(1, 2, 3) == 6