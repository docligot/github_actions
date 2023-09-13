import requests
import getpass
import json
import os
import time
import logging

from .src.data_extraction.nasa_appeears import get_earthdata_token, check_or_get_token, read_geojson, create_task_payload, create_task, wait_for_task_completion, download_files, fetch_nasa_appeears

def test_get_earthdata_token(): 
    assert get_earthdata_token() == None

def test_check_or_get_token(): 
    assert check_or_get_token(1) == 1
    
def test_read_geojson():
    assert read_geojson(1, 2, 3) == 6

def test_create_task_payload():
    assert create_task_payload(1, 2, 3, 4, 5) == 15
    
def test_create_task():
    assert create_task(1, 2, 3, 4, 5, 6) == 21
'''
def test_wait_for_task_completion():
    assert wait_for_task_completion(1, 2) == 3
    
def test_download_files():
    assert download_files(1, 2, 3) == 6
    
def test_download_files():
    assert download_files(1, 2, 3) == 6
    
def test_fetch_nasa_appeears(): 
    assert fetch_nasa_appeears(1, 2, 3, 4, 5, 6) == 21
'''