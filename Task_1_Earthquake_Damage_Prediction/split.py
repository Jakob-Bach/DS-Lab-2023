"""Split earthquake data for course-internal scoring

Script that creates and saves a stratified train-test split of the earthquake data.
"""


import pathlib

import pandas as pd
import sklearn.model_selection


INPUT_DIR = pathlib.Path('data/')
OUTPUT_DIR = pathlib.Path('data/scoring/')
SEED = 25


if __name__ == '__main__':
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X = pd.read_csv(INPUT_DIR / 'train_values.csv')
    y = pd.read_csv(INPUT_DIR / 'train_labels.csv')

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y['damage_grade'])

    X_train.to_csv(OUTPUT_DIR / 'train_values.csv', index=False, lineterminator='\n')
    X_test.to_csv(OUTPUT_DIR / 'test_values.csv', index=False, lineterminator='\n')
    y_train.to_csv(OUTPUT_DIR / 'train_labels.csv', index=False, lineterminator='\n')
    y_test.to_csv(OUTPUT_DIR / 'test_labels.csv', index=False, lineterminator='\n')
