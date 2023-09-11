def test_nasa_worldview():
    import requests
    import wget
    import datetime
    import os
    import json
    import time
    from datetime import timedelta

    from .src.data_extraction.nasa_worldview import fetch_nasa_worldview