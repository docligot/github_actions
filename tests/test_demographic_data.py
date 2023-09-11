import os
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.windows import from_bounds
from rasterio.warp import transform as rasterio_transform
from shapely.geometry import Point
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

from .src.data_preparation.demographic_data import process_demogs

def test_demographic_data():
    process_demogs("PHL", 2)
