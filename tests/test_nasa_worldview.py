import requests
import wget
import datetime
import os
import json
import time
from datetime import timedelta

from .src.data_extraction.nasa_worldview import get_lat_lon, latlong_gen, url_gen, next_date, fetch_nasa_worldview


def test_get_lat_lon():
    assert get_lat_lon(1) == 1
'''
def test_latlong_gen():
    assert latlong_gen(1, 2) == 3

def test_url_gen():
    assert url_gen(1, 2, 3, 4, 5, 6) == 21
    
def test_next_date():
    assert next_date(1, 2) == 3
    
def test_fetch_nasa_worldview():
    assert fetch_nasa_worldview(1, 2, 3, 4, 5, 6) == 21
'''