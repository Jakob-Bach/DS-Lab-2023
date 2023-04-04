"""Course-internal scoring on earthquake data

Script that finds all submission files in some directory, checks their validity, and scores them
against the ground truth.
"""


import csv
import pathlib

import pandas as pd
import sklearn.metrics


DATA_DIR = pathlib.Path('data/scoring/')  # needs to contain submissions and ground truth solution
METRIC = sklearn.metrics.matthews_corrcoef  # competition uses "accuracy_score"


def validate_submission(submission: pd.DataFrame, ground_truth: pd.DataFrame) -> str:
    if submission.shape[0] != ground_truth.shape[0]:
        return 'Number of predictions wrong (could also be issue with header).'
    if submission.shape[1] != ground_truth.shape[1]:
        return 'Number of columns wrong (index column might be saved).'
    if list(submission) != list(ground_truth):
        return 'At least one column name wrong (might be quoted).'
    if submission.isna().any().any():
        return 'At least one NA.'
    if not submission['damage_grade'].isin([1, 2, 3]).all():
        return 'At least one invalid class label.'
    return 'Valid.'


def score_submission(submission: pd.DataFrame, ground_truth: pd.DataFrame) -> float:
    submission = submission.merge(ground_truth, on='building_id')
    return METRIC(y_true=submission['damage_grade_y'], y_pred=submission['damage_grade_x'])


if __name__ == '__main__':
    results = []
    ground_truth = pd.read_csv(DATA_DIR / 'test_labels.csv')
    submission_files = list(DATA_DIR.glob('*_prediction.csv'))
    for submission_file in submission_files:
        submission = pd.read_csv(submission_file, sep=',', quoting=csv.QUOTE_NONE, header=0,
                                 escapechar=None, encoding='utf-8')
        team_name = submission_file.stem.replace('_prediction', '')
        validity_status = validate_submission(submission=submission, ground_truth=ground_truth)
        if validity_status == 'Valid.':
            score = score_submission(submission=submission, ground_truth=ground_truth)
        else:
            score = float('nan')
        results.append({'Team': team_name, 'Score': score, 'Validity': validity_status})
    results = pd.DataFrame(results).sort_values(by='Score', ascending=False)
    print(results.round(2), end='\n\n')
