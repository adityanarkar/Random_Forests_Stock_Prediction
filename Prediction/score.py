def get_score(selector, X_test, y_test):
    predictions = selector.predict(X_test)
    score = 0
    for i in range(0, len(X_test)):
        if predictions[i] == y_test[i]:
            score += 1
    return score
