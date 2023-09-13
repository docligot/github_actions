import os
import osmnx as ox
import geopandas as gpd
import pandas as pd
from datetime import datetime
from retrying import retry

# Update this line to use the new settings module
ox.settings.timeout = 60

from .src.data_extraction.osm import retry_if_file_not_found_error, create_directory, fetch_osm

def test_retry_if_file_not_found_error():
    assert retry_if_file_not_found_error(1) == 1

def test_create_directory():
    assert create_directory(1) == 1

def test_fetch_osm():
    assert fetch_osm(1, 2) == 3

