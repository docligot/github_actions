import requests
import getpass
import json
import os
import time
import logging

from .src.data_extraction.nasa_appeears import fetch_nasa_appeears

def test_nasa_appeears():
    fetch_nasa_appeears(headers, "PHL", "Zamboanga Sibugay", 2, '01-01-2021', '01-10-2021')