from pytrends.request import TrendReq
import pandas as pd
import time
import os
import random
import logging

from .src.data_extraction.google_trends import fetch_google_trends

def test_google_trends():
    fetch_google_trends('PHL', '2016-01-10', '2021-01-10')
