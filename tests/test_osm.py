def test_osm():
    import os
    import osmnx as ox
    import geopandas as gpd
    import pandas as pd
    from datetime import datetime
    from retrying import retry

    # Update this line to use the new settings module
    ox.settings.timeout = 60

    from .src.data_extraction.osm import fetch_osm
