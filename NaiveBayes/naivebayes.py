from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import numpy as np
import data_preparation as dp

df, data_to_predict = dp.data_preparation('../data/AAPL.csv', 10).data_frame_with_features()

    # convert dataframe to numpy array
data = df.to_numpy()
X = np.asarray(list(map(lambda row: row[:-1], data)))
y = np.asarray(list(map(lambda row: row[-1], data)))

# training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
score = gnb.score(X_test, y_test)
print(score)
print(gnb.predict(data_to_predict))