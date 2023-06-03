"""Predict with decision tree for SAT-solving data

Script that train a simple decision tree for both prediction targets of the SAT-solving data.
"""


import pathlib

import pandas as pd
import sklearn.tree

import split


DATA_DIR = pathlib.Path('data/')


if __name__ == '__main__':
    for target in split.TARGETS:
        X_train = pd.read_csv(DATA_DIR / f'{target}_X_train.csv')
        X_test = pd.read_csv(DATA_DIR / f'{target}_X_test.csv')
        y_train = pd.read_csv(DATA_DIR / f'{target}_y_train.csv')

        model = sklearn.tree.DecisionTreeClassifier(random_state=25)
        model.fit(X=X_train.drop(columns='hash'), y=y_train[target])
        y_test_pred = model.predict(X=X_test.drop(columns='hash'))
        y_test_pred = pd.DataFrame({'hash': X_test['hash'], target: y_test_pred})

        y_test_pred.to_csv(DATA_DIR / f'{target}_tree_prediction.csv', index=False)
