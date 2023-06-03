"""Predict majority for SAT-solving data

Script that creates a simple baseline for both prediction targets of the SAT-solving data by
predicting the majority class.
"""


import pathlib

import pandas as pd
import sklearn.dummy

import split


DATA_DIR = pathlib.Path('data/')


if __name__ == '__main__':
    for target in split.TARGETS:
        X_test = pd.read_csv(DATA_DIR / f'{target}_X_test.csv')
        y_train = pd.read_csv(DATA_DIR / f'{target}_y_train.csv')

        model = sklearn.dummy.DummyClassifier(strategy='most_frequent')
        model.fit(X=None, y=y_train[target])
        y_test_pred = model.predict(X=X_test)
        y_test_pred = pd.DataFrame({'hash': X_test['hash'], target: y_test_pred})

        y_test_pred.to_csv(DATA_DIR / f'{target}_majority_prediction.csv', index=False)
