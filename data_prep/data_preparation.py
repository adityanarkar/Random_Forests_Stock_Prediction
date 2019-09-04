import pandas as pd
from data_prep import features


class data_preparation(object):

    def __init__(self, filepath: str, window_size: int):
        self.filepath = filepath
        self.window = window_size

    def get_data(self, filepath: str) -> pd.DataFrame:
        return pd.read_csv(filepath)

    def create_label(self, row):
        row['class'] = 1 if row['adjusted_close'] < row['shifted_value'] else -1
        return row

    def get_fresh_data_for_prediction(self, df: pd.DataFrame):
        result = df.where(df['shifted_value'].isna())
        result.dropna(thresh=1, inplace=True)
        return result

    def data_frame_with_features(self):
        df = self.get_data(self.filepath)

        df.drop(columns=["timestamp"], inplace=True)
        df.dropna(inplace=True)

        features.simpleMA(df, self.window, True)
        features.weightedMA(df, self.window, True)
        features.EMA(df, self.window, True)
        features.momentum(df, self.window)
        features.stochasticK(df, self.window, True)
        features.stochasticD(df, self.window, True)
        features.MACD(df, True)
        features.RSI(df)
        features.williamsR(df, 9, True)
        features.ADIndicator(df)
        features.diff_n_Months(df, 90)
        features.diff_current_lowest_low(df, 90)
        features.diff_current_highest_high(df, 90)
        # features.CCI(df, 20)

        df['shifted_value'] = df['adjusted_close'].shift(-1 * self.window)
        data_to_predict = self.get_fresh_data_for_prediction(df)
        df = df.apply(lambda x: self.create_label(x), axis=1)
        df.dropna(inplace=True)
        df.drop(columns=['shifted_value', 'dividend_amount', 'split_coefficient'], inplace=True)
        data_to_predict.drop(columns=['shifted_value', 'dividend_amount', 'split_coefficient'], inplace=True)
        return df, data_to_predict