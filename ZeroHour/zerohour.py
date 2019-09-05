import numpy as np
import pandas as pd
import collections


class zeroHour(object):
    def __init__(self, prediction_class):
        self.prediction_class = prediction_class

    def predict(self, y):
        if self.prediction_class == 1:
            return np.ones(len(y))
        else:
            return np.full(len(y), -1)

    def fit(self, data: pd.DataFrame):
        ups = collections.Counter(data).get(1)
        downs = collections.Counter(data).get(-1)
        self.prediction_class = 1 if ups >= downs else -1