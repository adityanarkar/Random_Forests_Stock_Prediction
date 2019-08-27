import requests
import time
import pandas as pd

api_calls = 0
key = "YOUR_API_KEY"

f = open('TICKR.txt', 'r+')
lines = f.readlines()
lines = list(map(lambda x: x.replace('\n', ''), lines))

for sym in lines:
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={sym}&outputsize=full&interval=5min&apikey={key}=csv"
    filepath = f"data/{sym}.csv"

    r = requests.get(url=url)
    api_calls += 1

    with open(filepath, 'wb') as f:
        f.write(r.content)

    df = pd.read_csv(filepath)
    df = df[::-1]
    print(filepath)
    print(df.head())
    df.to_csv(filepath, index=False)

    if api_calls % 5 == 0:
        time.sleep(90)
    else:
        time.sleep(5)
