import os
import geopandas as gpd
import pytest
import os
import json
import requests
import time
from fiona.crs import from_epsg
from time import sleep
import warnings

warnings.filterwarnings("ignore")

from .admin_boundaries import fetch_geoboundaries, save_admin_regions

def test_usage():
    assert save_admin_regions("PHL", 2) is None
