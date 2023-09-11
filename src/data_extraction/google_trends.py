#!/usr/bin/env python
# coding: utf-8

# In[2]:


from pytrends.request import TrendReq
import pandas as pd
import time
import os
import random
import logging

def fetch_google_trends(iso_country_code, start_date, end_date):
    # Initialize logging
    logging.basicConfig(filename='fetch_google_trends.log', level=logging.INFO)
    
    # Import ISO 3166-2 and Region CSV
    csv_file_path = os.path.join(os.path.dirname(os.getcwd()),'data', f'Subdivision_{iso_country_code}.csv')
    iso_df = pd.read_csv(csv_file_path)
    
    # Start Payload
    pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 50), retries=3, backoff_factor=0.5)
    
    # Initialize DataFrame to store Google Trends data
    df_trends = pd.DataFrame()
    
    for index, row in iso_df.iterrows():
        geo = row['ISO 3166-2']
        region = row['Region']
        
        try:
            # Get related queries
            pytrends.build_payload(["Dengue"], timeframe=f'{start_date} {end_date}', geo=geo, gprop='')
            rel_queries = pytrends.related_queries()

            if rel_queries and 'Dengue' in rel_queries:
                dengue_data = rel_queries['Dengue']
                rising_queries = dengue_data.get("rising")
                top_queries = dengue_data.get("top")

                rising_list = rising_queries["query"].tolist() if rising_queries is not None else []
                top_list = top_queries["query"].tolist() if top_queries is not None else []

                kw_list = ["dengue"] + rising_list + top_list
                kw_list = list(set(kw_list))
            else:
                kw_list = ["dengue"]

            for kw in kw_list:
                pytrends.build_payload(kw_list=[kw], timeframe=f'{start_date} {end_date}', geo=geo, gprop='')
                search_vol = pytrends.interest_over_time().reset_index()
                search_vol = search_vol.rename(columns={kw: "value"})
                search_vol["keyword"] = kw
                search_vol["geo"] = geo
                search_vol["region"] = region
                search_vol["date_extracted"] = pd.Timestamp.now().strftime('%Y-%m-%d')

                df_trends = df_trends.append(search_vol, ignore_index=True)

        except Exception as e:
            logging.error(f"Error occurred for geo {geo}: {e}")
            time.sleep(60)  # Sleep for 60 seconds before retrying
            continue  # Skip to the next iteration
        
        print('Scraping done for:', geo)
        
        # Random sleep between 10 to 20 seconds
        time.sleep(random.randint(10, 20))
    
    # Save DataFrame as CSV
    save_path = os.path.join(os.path.dirname(os.getcwd()),'data', 'Google Trends', f"{iso_country_code}.csv")
    df_trends.to_csv(save_path, index=False)

# Example usage
#fetch_google_trends('PHL', '2016-01-10', '2021-01-10')


# In[ ]:




