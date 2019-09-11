import os
import numpy as np
import requests
import time
import pandas as pd


def collect_data(tickrs_file):
    api_calls = 0
    key = "YOUR_KEY_HERE"
    f = open(tickrs_file, 'r+')
    lines = f.readlines()
    lines = list(map(lambda x: x.replace('\n', ''), lines))
    dirname = os.path.dirname(__file__)

    for sym in lines:
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={sym}&outputsize=full&apikey={key}&datatype=csv"
        filepath = os.path.join(dirname, f"../data/{sym}.csv")

        r = requests.get(url=url)
        api_calls += 1

        with open(filepath, 'wb') as f:
            f.write(r.content)

        df = pd.read_csv(filepath)
        df = df[::-1]
        print(filepath)
        df.to_csv(filepath, index=False)

        if api_calls % 5 == 0:
            time.sleep(90)
        else:
            time.sleep(5)


def sample_data(no_of_symbols: int, filepath: str):
    file = pd.read_csv('data/More_data/companylist.csv')
    file = file["Symbol"].apply(lambda x: x if not "^" in x and not "." in x else np.nan)
    file.dropna(inplace=True)
    file = file.sample(n=no_of_symbols)
    file.to_csv(filepath, index=False, header=False)
    return file
