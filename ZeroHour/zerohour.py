import numpy as np
import pandas as pd
import collections


class zeroHour(object):
    prediction_class = 0

    def predict(self, x):
        if self.prediction_class == 1:
            return np.ones(len(x))
        else:
            return np.full(len(x), -1)

    def fit(self, data: pd.DataFrame):
        pred_class = data[:, -1]
        ups = collections.Counter(pred_class).get(1)
        downs = collections.Counter(pred_class).get(-1)
        self.prediction_class = 1 if ups >= downs else -1
