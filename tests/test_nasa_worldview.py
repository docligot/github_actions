import requests
import wget
import datetime
import os
import json
import time
from datetime import timedelta

from .src.data_extraction.nasa_worldview import fetch_nasa_worldview

def test_nasa_worldview():
    fetch_nasa_worldview("PHL", "Zamboanga Sibugay", "Zamboanga Peninsula", "2016-01-10", "2021-01-10", "W")
