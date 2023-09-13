import requests
import re
import os
from .src.data_extraction.demographics import fetch_relative_wealth_index, download_rwi, fetch_population_density, download_popmap
    
def test_fetch_relative_wealth_index():
    assert fetch_relative_wealth_index(1, 2) == 3

def test_download_rwi():
    assert download_rwi(1, 2, 3) == 6
    
def test_fetch_population_density():
    assert fetch_population_density(1, 2, 3) == 6

def test_download_popmap():
    assert download_popmap(1, 2, 3, 4) == 10