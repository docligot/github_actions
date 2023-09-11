def test_functions():
    import os
    import json
    import requests
    import geopandas as gpd
    from fiona.crs import from_epsg
    from time import sleep
    import warnings
    warnings.filterwarnings("ignore")
    from .src.data_extraction.admin_boundaries import fetch_geoboundaries, save_admin_regions
