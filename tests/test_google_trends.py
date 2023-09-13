
from pytrends.request import TrendReq
import pandas as pd
import time
import os
import random
import logging

from .src.data_extraction.google_trends import fetch_google_trends

def test_fetch_google_trends():
    assert fetch_google_trends(1, 2, 3) == 6

