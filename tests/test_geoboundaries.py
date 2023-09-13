import os
import json
import requests
import geopandas as gpd
from fiona.crs import from_epsg
from time import sleep
import warnings
warnings.filterwarnings("ignore")
from .src.data_extraction.admin_boundaries import fetch_geoboundaries, save_admin_regions

def test_fetch_geoboundaries():
    assert fetch_geoboundaries(1, 2) == 3

def test_save_admin_regions():
    assert save_admin_regions(1, 2) == 3