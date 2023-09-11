def test_nasa_appeears():
    import requests
    import getpass
    import json
    import os
    import time
    import logging

    from .src.data_extraction.nasa_appeears import fetch_nasa_appeears, get_earthdata_token
