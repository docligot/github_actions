import os
import geopandas as gpd
import pytest
import os
import json
import requests
import geopandas as gpd
from fiona.crs import from_epsg
from time import sleep
import warnings

warnings.filterwarnings("ignore")

from .admin_boundaries import Admin_boundaries

# Define test cases for the fetch_geoboundaries function
@pytest.mark.parametrize("iso_country_code, admin_level", [
    ("US", 1),  # Test with valid country code and admin level
    ("INVALID", 1),  # Test with invalid country code
    ("US", 10),  # Test with invalid admin level
])
def test_fetch_geoboundaries(iso_country_code, admin_level):
    geoBoundary = Admin_boundaries.fetch_geoboundaries(iso_country_code, admin_level)

    if iso_country_code == "INVALID" or admin_level > 5:
        # For invalid inputs, the function should return None
        assert geoBoundary is None
    else:
        # For valid inputs, the function should return a GeoDataFrame
        assert isinstance(geoBoundary, gpd.GeoDataFrame)
        # Check that the GeoDataFrame has some rows
        assert len(geoBoundary) > 0

# Define test cases for the save_admin_regions function
@pytest.mark.parametrize("iso_country_code, admin_level", [
    ("US", 1),  # Test with valid inputs
])
def test_save_admin_regions(iso_country_code, admin_level, tmpdir):
    geoBoundary = Admin_boundaries.fetch_geoboundaries(iso_country_code, admin_level)
    
    if geoBoundary is not None:
        # Create a temporary directory for testing
        temp_dir = tmpdir.mkdir("test_data")
        
        # Modify the data_folder path to use the temporary directory
        data_folder = os.path.join(temp_dir, 'GeoJSON', f"{iso_country_code}/ADM{admin_level}")
        
        # Call the save_admin_regions function with the temporary directory
        save_admin_regions(iso_country_code, admin_level, data_folder)
        
        # Check if GeoJSON files were created in the temporary directory
        assert len(os.listdir(data_folder)) > 0

if __name__ == "__main__":
    pytest.main()
