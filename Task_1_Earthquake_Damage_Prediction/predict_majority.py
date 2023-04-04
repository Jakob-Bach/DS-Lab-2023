"""Predict majority for earthquake data

Script that creates a simple baseline for the earthquake data by predicting the majority class.
"""


import pathlib

import pandas as pd
import sklearn.dummy


DATA_DIR = pathlib.Path('data/scoring/')


if __name__ == '__main__':
    X_test = pd.read_csv(DATA_DIR / 'test_values.csv')
    y_train = pd.read_csv(DATA_DIR / 'train_labels.csv')

    model = sklearn.dummy.DummyClassifier(strategy='most_frequent')
    model.fit(X=None, y=y_train['damage_grade'])
    y_test_pred = model.predict(X=X_test)
    y_test_pred = pd.DataFrame({'building_id': X_test['building_id'], 'damage_grade': y_test_pred})

    y_test_pred.to_csv(DATA_DIR / 'majority_prediction.csv', index=False)
