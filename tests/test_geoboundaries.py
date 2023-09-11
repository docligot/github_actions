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

from .admin_boundaries import Admin_boundaries

def test_usage():
    assert Admin_boundaries.save_admin_regions("PHL", 2) != null
