import numpy as np
import pandas as pd


def simpleMA(df: pd.DataFrame, moving_avg_window, discretize: bool):
    df['SMA'] = df['adjusted_close'].rolling(window=moving_avg_window).mean()
    df.dropna(inplace=True)
    if discretize:
        df["SMA"] = (df['adjusted_close'] > df['SMA']).apply(lambda x: 1 if x else -1)


def weighted_calculations(x, moving_avg_window):
    wts = np.arange(start=1, stop=moving_avg_window + 1, step=1)
    return (wts * x).mean()


def weightedMA(df: pd.DataFrame, moving_avg_window, discretize: bool):
    df['WMA'] = df['adjusted_close'] \
        .rolling(window=moving_avg_window) \
        .apply(lambda x: weighted_calculations(x, moving_avg_window))
    df.dropna(inplace=True)
    if discretize:
        df["WMA"] = (df['adjusted_close'] > df['WMA']).apply(lambda x: 1 if x else -1)


def EMA(df: pd.DataFrame, moving_avg_window, discretize: bool):
    df[f"{moving_avg_window}-day-EMA"] = df['adjusted_close'] \
        .ewm(span=moving_avg_window, adjust=False) \
        .mean()
    df.dropna(inplace=True)
    if discretize:
        df[f"{moving_avg_window}-day-EMA"] = (df['adjusted_close'] > df[f"{moving_avg_window}-day-EMA"]).apply(
            lambda x: 1 if x else -1)


def momentum(df: pd.DataFrame, moving_window):
    df['Momentum'] = df['adjusted_close'].rolling(window=moving_window).apply(lambda x: x[0] - x[-1])
    df['Momentum'] = df['Momentum'].rolling(window=2).apply(lambda x: 1 if x[1] > x[0] else -1)
    df.dropna(inplace=True)
    return df


def stochasticK_calculations(x):
    currentClose = x["adjusted_close"]
    highestHigh = x["highestHigh"]
    lowestLow = x["lowestLow"]
    high_low = (highestHigh - lowestLow)
    return (currentClose - lowestLow) / high_low


def stochasticK(df: pd.DataFrame, moving_window, discretize: bool):
    df["highestHigh"] = df["high"].rolling(window=moving_window).apply(lambda x: max(x))
    df["lowestLow"] = df["low"].rolling(window=moving_window).apply(lambda x: min(x))
    df.dropna(inplace=True)
    df["StochasticK"] = df[["adjusted_close", "highestHigh", "lowestLow"]] \
        .apply(lambda x: stochasticK_calculations(x), axis=1)
    df.drop(columns=["highestHigh", "lowestLow"], inplace=True)
    if discretize:
        df["StochasticK"] = df["StochasticK"].rolling(window=2).apply(lambda x: discretizeOscillator(x))
    df.dropna(inplace=True)


def stochasticD(df: pd.DataFrame, moving_window, discretize: bool):
    if "StochasticK" in df:
        df["StochasticD"] = df["StochasticK"].rolling(3).mean()
        df.dropna(inplace=True)
        if discretize:
            df["StochasticD"] = df["StochasticD"].rolling(window=2).apply(lambda x: discretizeOscillator(x))
        df.dropna(inplace=True)
    else:
        return stochasticD(stochasticK(df, moving_window), moving_window, discretize)


def RSI(df: pd.DataFrame):
    df["GL"] = df["adjusted_close"].rolling(window=2).apply(lambda x: x[0] - x[1])
    df.dropna(inplace=True)
    df["RSI"] = df["GL"].rolling(window=14).apply(lambda x: calculateRSI(x))
    df.drop(columns=["GL"], inplace=True)
    df.dropna(inplace=True)
    print(df.tail())
    return df


def calculateRSI(x):
    avgGain = abs(x[x > 0].sum() / 14)
    avgLoss = abs(x[x < 0].sum() / 14)
    RS = avgGain / avgLoss
    result = 100 - (100 / (1 + RS))
    return result


def MACD(df: pd.DataFrame, discretize):
    EMA(df, 9, False)
    EMA(df, 12, False)
    EMA(df, 26, False)
    df.dropna(inplace=True)
    df['MACD'] = np.nan
    MACDLine = df['12-day-EMA'] - df['26-day-EMA']
    df['MACD'] = MACDLine - df['9-day-EMA']
    df.dropna(inplace=True)

    if discretize:
        df["MACD"] = df["MACD"].rolling(window=2).apply(lambda x: discretizeOscillator(x))

    df.dropna(inplace=True)


def calculateWilliamsR(x):
    highestHigh = x["highestHigh"]
    lowestLow = x["lowestLow"]
    adjusted_close = x["adjusted_close"]
    return (highestHigh - adjusted_close) / (highestHigh - lowestLow)


def discretizeOscillator(x):
    return 1 if x[1] > x[0] else -1


def williamsR(df: pd.DataFrame, lookback_period: int, discretize: bool):
    df["highestHigh"] = df["high"].rolling(window=lookback_period).max()
    df["lowestLow"] = df["low"].rolling(window=lookback_period).max()
    df.dropna(inplace=True)
    df["williamsR"] = df[["highestHigh", "lowestLow", "adjusted_close"]].apply(lambda x: calculateWilliamsR(x), axis=1)
    df.drop(columns=["highestHigh", "lowestLow"], inplace=True)
    if discretize:
        df["williamsR"] = df["williamsR"].rolling(window=2).apply(lambda x: discretizeOscillator(x))
        df.dropna(inplace=True)


def ADIndicator(df: pd.DataFrame):
    df["AD"] = ((df["adjusted_close"] - df["low"]) - (df["high"] - df["adjusted_close"]) / (df["high"] - df["low"])) * \
               df["volume"]
    df["AD"] = df["AD"].cumsum()
    df["AD"] = df["AD"].rolling(window=2).apply(lambda x: discretizeOscillator(x))
    df.dropna(inplace=True)


def CCI(df: pd.DataFrame, window):
    df["TP"] = (df["high"] + df["low"] + df["adjusted_close"]) / 3
    df[f"{window}-day-SMA-TP"] = df["TP"].rolling(window=window).mean()
    df.dropna(inplace=True)

    df[f"{window}-day-mean-deviation"] = abs(df["TP"] - df[f"{window}-day-SMA-TP"])
    df["CCI"] = (df["TP"] - df[f"{window}-day-SMA-TP"]) / (0.15 * df[f"{window}-day-mean-deviation"])
    print(df.head())
    df.drop(inplace=True, columns=[f"{window}-day-SMA-TP", f"{window}-day-mean-deviation", "TP"])


def diff_n_Months(df: pd.DataFrame, n):
    df["diff_3_months"] = df["adjusted_close"].rolling(window=n).apply(lambda x: (x[0] - x[-1]) / x[-1])
    df.dropna(inplace=True)


def checkValue(value):
    if value >= 0:
        return 1
    else:
        return -1
