"""Course-internal scoring on SAT-solving data

Script that finds all submission files in some directory, checks their validity, and scores them
against a ground truth (for both prediction targets).
"""


import csv
import pathlib

import pandas as pd
import sklearn.metrics

import split


DATA_DIR = pathlib.Path('data/scoring/')  # needs to contain submissions and ground truth solution


def validate_submission(submission: pd.DataFrame, ground_truth: pd.DataFrame, target: str) -> str:
    if submission.shape[0] != ground_truth.shape[0]:
        return 'Number of predictions wrong (could also be issue with header).'
    if submission.shape[1] != ground_truth.shape[1]:
        return 'Number of columns wrong (index column might be saved).'
    if list(submission) != list(ground_truth):
        return 'At least one column name wrong (might be quoted).'
    if submission.isna().any().any():
        return 'At least one NA.'
    if submission[target].dtype == 'int64':
        return 'Predicted class labels are integer.'
    return 'Valid.'


if __name__ == '__main__':
    for target in split.TARGETS:
        print('Target:', target)
        results = []
        ground_truth = pd.read_csv(DATA_DIR / f'{target}_y_test.csv')
        submission_files = list(DATA_DIR.glob(f'{target}_*_prediction.csv'))
        for submission_file in submission_files:
            submission = pd.read_csv(submission_file, sep=',', quoting=csv.QUOTE_NONE, header=0,
                                     escapechar=None, encoding='utf-8')
            team_name = submission_file.stem.replace(f'{target}_', '').replace('_prediction', '')
            validity_status = validate_submission(submission=submission, ground_truth=ground_truth,
                                                  target=target)
            if validity_status == 'Valid.':
                submission = submission.merge(ground_truth, on='hash')
                score = sklearn.metrics.matthews_corrcoef(
                    y_true=submission[f'{target}_y'], y_pred=submission[f'{target}_x'])
            else:
                score = float('nan')
            results.append({'Team': team_name, 'Score': score, 'Validity': validity_status})
        results = pd.DataFrame(results).sort_values(by='Score', ascending=False)
        print(results.round(2), end='\n\n')
