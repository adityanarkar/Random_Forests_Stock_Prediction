from functools import reduce
import random_forests_stock_prediction as rf

n_estimators = [100, 150, 200]
max_depth = [7, 8, 9, 10]
random_state = [0, 1, 2]
combinations = []


def get_score(best_combo: tuple, combination):
    arg_estimator = combination['estimator']
    arg_depth = combination['depth']
    arg_state = combination['state']
    score = rf.random_forest_classifier(arg_estimator, arg_depth, arg_state)
    if score > best_combo[0]:
        return tuple((score, combination))
    return best_combo


for estimator in n_estimators:
    for depth in max_depth:
        for state in random_state:
            combinations.append({'estimator': estimator, 'depth': depth, 'state': state})

t = reduce(lambda a, x: get_score(a, x), combinations, tuple((0, {})))
print(t)
