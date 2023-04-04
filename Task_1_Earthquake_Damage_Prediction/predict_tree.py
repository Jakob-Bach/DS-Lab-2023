"""Predict with decision tree for earthquake data

Script that train a simple decision tree for the earthquake data.
"""


import pathlib

import numpy as np
import pandas as pd
import sklearn.preprocessing
import sklearn.tree


DATA_DIR = pathlib.Path('data/scoring/')


if __name__ == '__main__':
    # Load
    X_train = pd.read_csv(DATA_DIR / 'train_values.csv')
    X_test = pd.read_csv(DATA_DIR / 'test_values.csv')
    y_train = pd.read_csv(DATA_DIR / 'train_labels.csv')

    # Preprocess (one-hot encode categorical features with a cap on number of encoding features):
    test_ids = X_test['building_id']
    multi_category_features = [col for col in X_train.columns if X_train[col].dtype == 'object']
    multi_category_features.extend([f'geo_level_{i}_id' for i in (1, 2, 3)])
    encoder = sklearn.preprocessing.OneHotEncoder(sparse_output=False, max_categories=40,
                                                  handle_unknown='infrequent_if_exist')
    encoder.fit(X=X_train[multi_category_features])
    X_train = np.concatenate([
        X_train.drop(columns=(multi_category_features + ['building_id'])).values,
        encoder.transform(X=X_train[multi_category_features])], axis=1)
    X_test = np.concatenate([
        X_test.drop(columns=(multi_category_features + ['building_id'])).values,
        encoder.transform(X=X_test[multi_category_features])], axis=1)

    # Predict
    model = sklearn.tree.DecisionTreeClassifier(random_state=25)
    model.fit(X=X_train, y=y_train['damage_grade'])
    y_test_pred = model.predict(X=X_test)
    y_test_pred = pd.DataFrame({'building_id': test_ids, 'damage_grade': y_test_pred})

    # Save
    y_test_pred.to_csv(DATA_DIR / 'tree_prediction.csv', index=False)
