import requests
import pandas as pd
import io
import time
import random                                     
import os
import json

from .src.data_extraction.meteorological import fetch_weather_data

def test_meteorological():
    fetch_weather_data("PHL", 2016, 2021)
