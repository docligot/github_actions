def test_google_trends():
    from pytrends.request import TrendReq
    import pandas as pd
    import time
    import os
    import random
    import logging

    from .src.data_extraction.google_trends import fetch_google_trends

