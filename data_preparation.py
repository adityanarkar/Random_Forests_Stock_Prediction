import pandas as pd
import features


class data_preparation(object):

    def __init__(self, filepath: str, window_size: int):
        self.filepath = filepath
        self.window = window_size

    def get_data(self, filepath: str) -> pd.DataFrame:
        return pd.read_csv(filepath)

    def create_label(self, row):
        row['class'] = 1 if row['Adj Close'] < row['shifted_value'] else -1
        return row

    def get_fresh_data_for_prediction(self, df: pd.DataFrame):
        result = df.where(df['shifted_value'].isna())
        result.dropna(thresh=1, inplace=True)
        result.drop(columns=['shifted_value'], inplace=True)
        return result

    def data_frame_with_features(self):
        df = self.get_data(self.filepath)

        df.drop(columns=["Date"], inplace=True)
        df.dropna(inplace=True)

        features.simpleMA(df, self.window, True)
        features.weightedMA(df, self.window)
        features.EMA(df, self.window)
        features.momentum(df, self.window)
        features.stochasticK(df, self.window)
        features.stochasticD(df, self.window)
        features.MACD(df)
        features.RSI(df)

        df['shifted_value'] = df['Adj Close'].shift(-10)
        data_to_predict = self.get_fresh_data_for_prediction(df)
        df = df.apply(lambda x: self.create_label(x), axis=1)
        df.dropna(inplace=True)
        df.drop(columns=['shifted_value'], inplace=True)

        # self.to_arff(df, "arff/file.arff", "relation")
        # df.to_csv("file.csv", index=False)

        return df, data_to_predict

    def to_arff(self, df: pd.DataFrame, arff_file_path: str, relation: str):
        result = f"@RELATION {relation}\n"
        headers = df.columns
        for header in headers:
            if header in ["timestamp", "date"]:
                result += f"@ATTRIBUTE {header} date\n"
            elif header == "class":
                result += "@ATTRIBUTE class {1, -1}\n"
            else:
                result += f"@ATTRIBUTE {header} NUMERIC\n"

        result += "@DATA\n"

        x = df.to_string(header=False,
                         index=False,
                         index_names=False).split('\n')

        rows = [','.join(ele.split()) for ele in x]

        for data in rows:
            result += f"{data}\n"

        with open(arff_file_path, 'w') as f:
            f.write(result)
