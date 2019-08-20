import numpy as np
import pandas as pd


def simpleMA(df: pd.DataFrame, moving_avg_window, discritize: bool):
    df['SMA'] = df['Adj Close'].rolling(window=moving_avg_window).mean()
    df.dropna(inplace=True)
    if discritize:
        return discritizeSMA(df)
    return df


def discritizeSMA(df: pd.DataFrame):
    df["SMA"] = np.where(df["Adj Close"] > df["SMA"], 1, 0)
    return df



def weighted_calculations(x, moving_avg_window):
    wts = np.arange(start=1, stop=moving_avg_window + 1, step=1)
    return (wts * x).mean()



def weightedMA(df: pd.DataFrame, moving_avg_window):
    df['WMA'] = df['Adj Close'] \
        .rolling(window=moving_avg_window) \
        .apply(lambda x: weighted_calculations(x, moving_avg_window))
    df.dropna(inplace=True)
    return df


def EMA(df: pd.DataFrame, moving_avg_window):
    df['EMA'] = df['Adj Close'] \
        .ewm(span=moving_avg_window, adjust=False) \
        .mean()
    df.dropna(inplace=True)
    return df


def discretizeMomentum(df: pd.DataFrame, row, prev):
    if prev != np.nan:
        if df.iloc[row, -1] < prev:
            prev = df.iloc[row, -1]
            df.iloc[row, -1] = -1
        else:
            prev = df.iloc[row, -1]
            df.iloc[row, -1] = 1
    return prev


def momentum(df: pd.DataFrame, moving_window):
    df['Momentum'] = df['Adj Close'].rolling(window=moving_window).apply(lambda x: x[0] - x[-1])
    df['Momentum'] = df['Momentum'].rolling(window=2).apply(lambda x: 1 if x[1] > x[0] else -1)
    df.dropna(inplace=True)
    return df


def stochasticK_calculations(x):
    print(x)


def stochasticK(df: pd.DataFrame, moving_window):
    # prev = np.nan
    # for row in range(window, len(df.index)):
    #     currentClose = df.iloc[row, close]
    #     highestHigh = df.iloc[row - window:row, high].max()
    #     lowestLow = df.iloc[row - window:row, low].min()
    #     df.iloc[row, -1] = (currentClose - lowestLow) / (highestHigh - lowestLow)
    #     prev = discretizeMomentum(df, row, prev)
    df.rolling_apply(lambda x: stochasticK_calculations(x))
    # highestHighs = df["High"].rolling(window=moving_window).apply(lambda x: max(x))
    # lowestLow = df["Low"].rolling(window=moving_window).apply(lambda x: min(x))
    # print(highestHighs)
    # print(lowestLow)


def stochasticD(df, K):
    prev = np.nan
    for row in range(2, len(df.index)):
        df.iloc[row, -1] = df.iloc[row - 2:row, K].mean()
        prev = discretizeMomentum(df, row, prev)


def RSI(df: pd.DataFrame, close):
    for row in range(15, len(df.index)):
        temp1 = df.iloc[row - 15:row - 2, close].reset_index(drop=True)
        temp2 = df.iloc[row - 14:row - 1, close].reset_index(drop=True)
        temp = temp1 - temp2
        AvgGain = temp[temp > 0].sum() / 14
        AvgLoss = -1 * (temp[temp < 0].sum()) / 14
        RS = AvgGain / AvgLoss
        RSIvalue = 100 - (100 / (1 + RS))
        if (RSIvalue > 70):
            df.iloc[row, -1] = -1
        elif (RSIvalue < 30):
            df.iloc[row, -1] = 1
        else:
            if (df.iloc[row - 1, -1] != np.nan and RSIvalue > df.iloc[row - 1, -1]):
                df.iloc[row, -1] = 1
            else:
                df.iloc[row, -1] = -1


def MACD(df: pd.DataFrame, close):
    EMA(df, 9, close)
    EMA(df, 12, close)
    EMA(df, 26, close)
    df.dropna(inplace=True)
    df['MACD'] = np.nan
    MACDLine = df['12-day-EMA'] - df['26-day-EMA']
    df['MACD'] = MACDLine - df['9-day-EMA']
    df['MACD'] = df['MACD'].diff()
    df.dropna(inplace=True)
    df['MACD'] = df['MACD'].apply(checkValue)


def checkValue(value):
    if value >= 0:
        return 1
    else:
        return -1
