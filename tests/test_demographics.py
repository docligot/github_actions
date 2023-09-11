import requests
import re
import os

from .src.data_extraction.demographics import fetch_relative_wealth_index, fetch_population_density

def test_population_density():
    fetch_population_density("Philippines", "PHL", "general")

def test_relative_wealth_index():
    fetch_relative_wealth_index("Philippines", "PHL")